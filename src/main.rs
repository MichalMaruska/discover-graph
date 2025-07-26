use petgraph::graph::{NodeIndex};
use petgraph::visit::{depth_first_search, DfsEvent, Control, IntoNeighbors, GraphBase, Visitable};
use petgraph::stable_graph::StableDiGraph;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::cell::RefCell;

/// Trait for providing vertices and edges on demand
pub trait GraphProvider<T> {
    /// Get neighbors of a given vertex
    fn get_neighbors(&mut self, vertex: &T) -> Vec<T>;

    /// Check if a vertex exists (optional validation)
    fn vertex_exists(&mut self, vertex: &T) -> bool {
        true // Default implementation assumes all vertices exist
    }
}

/// Dynamic graph wrapper that implements IntoNeighbors with on-demand discovery
pub struct DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    graph: StableDiGraph<T, ()>,
    vertex_to_node: HashMap<T, NodeIndex>,
    discovered: HashSet<T>,
    provider: RefCell<P>, // RefCell allows interior mutability
}

impl<T, P> DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    pub fn new(provider: P) -> Self {
        Self {
            graph: StableDiGraph::new(),
            vertex_to_node: HashMap::new(),
            discovered: HashSet::new(),
            provider: RefCell::new(provider),
        }
    }

    /// Get or create a node in the graph for the given vertex
    fn get_or_create_node(&mut self, vertex: T) -> NodeIndex {
        if let Some(&node_idx) = self.vertex_to_node.get(&vertex) {
            node_idx
        } else {
            let node_idx = self.graph.add_node(vertex.clone());
            self.vertex_to_node.insert(vertex, node_idx);
            node_idx
        }
    }

    /// Discover neighbors for a vertex and add them to the graph
    /// This is called by the IntoNeighbors implementation
    fn discover_if_needed(&self, node_idx: NodeIndex) {
        let vertex = &self.graph[node_idx];

        // Check if already discovered - we need to cast away const here
        // because we're doing lazy discovery in what appears to be a const context
        if self.discovered.contains(vertex) {
            return;
        }

        // This is safe because we're using RefCell for interior mutability
        let this = unsafe { &mut *(self as *const Self as *mut Self) };

        let neighbors = this.provider.borrow_mut().get_neighbors(vertex);
        this.discovered.insert(vertex.clone());

        println!("Dynamically discovering vertex {:?} with {} neighbors", vertex, neighbors.len());

        // Add neighbors to graph
        for neighbor in neighbors {
            if this.provider.borrow_mut().vertex_exists(&neighbor) {
                let neighbor_node = this.get_or_create_node(neighbor.clone());

                // Add edge if it doesn't exist
                if this.graph.find_edge(node_idx, neighbor_node).is_none() {
                    this.graph.add_edge(node_idx, neighbor_node, ());
                }
            }
        }
    }

    pub fn add_start_vertex(&mut self, vertex: T) -> Option<NodeIndex> {
        if self.provider.borrow_mut().vertex_exists(&vertex) {
            Some(self.get_or_create_node(vertex))
        } else {
            None
        }
    }

    pub fn get_vertex(&self, node_idx: NodeIndex) -> Option<&T> {
        self.graph.node_weight(node_idx)
    }

    pub fn get_graph(&self) -> &StableDiGraph<T, ()> {
        &self.graph
    }

    pub fn discovered_count(&self) -> usize {
        self.discovered.len()
    }

    pub fn edges_count(&self) -> usize {
        self.graph.edge_count()
    }
}

// Implement GraphBase for our wrapper
impl<T, P> GraphBase for DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    type NodeId = NodeIndex;
    type EdgeId = petgraph::graph::EdgeIndex;
}

// Implement Visitable for our wrapper
impl<T, P> Visitable for DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    type Map = <StableDiGraph<T, ()> as Visitable>::Map;

    fn visit_map(&self) -> Self::Map {
        self.graph.visit_map()
    }

    fn reset_map(&self, map: &mut Self::Map) {
        self.graph.reset_map(map)
    }
}

// Custom neighbors iterator that triggers discovery
pub struct DynamicNeighbors<'a, T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    graph: &'a DynamicGraph<T, P>,
    inner: petgraph::stable_graph::Neighbors<'a, (), petgraph::Directed>,
    node_idx: NodeIndex,
    discovered: bool,
}

impl<'a, T, P> Iterator for DynamicNeighbors<'a, T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        // Trigger discovery on first access
        if !self.discovered {
            self.graph.discover_if_needed(self.node_idx);
            self.discovered = true;

            // Get fresh iterator after discovery
            self.inner = self.graph.graph.neighbors(self.node_idx);
        }

        self.inner.next()
    }
}

// Implement IntoNeighbors for our wrapper - this is the key!
impl<'a, T, P> IntoNeighbors for &'a DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    type Neighbors = DynamicNeighbors<'a, T, P>;

    fn neighbors(self, node_idx: Self::NodeId) -> Self::Neighbors {
        DynamicNeighbors {
            graph: self,
            inner: self.graph.neighbors(node_idx),
            node_idx,
            discovered: false,
        }
    }
}

/// Main discoverer that uses the dynamic graph wrapper
pub struct GraphDiscoverer<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    dynamic_graph: DynamicGraph<T, P>,
}

impl<T, P> GraphDiscoverer<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    pub fn new(provider: P) -> Self {
        Self {
            dynamic_graph: DynamicGraph::new(provider),
        }
    }

    /// Perform DFS using petgraph's depth_first_search with dynamic discovery
    pub fn dfs_discover(&mut self, start: T) -> Vec<T> {
        let start_node = match self.dynamic_graph.add_start_vertex(start) {
            Some(node) => node,
            None => return Vec::new(),
        };

        let mut discovery_order = Vec::new();

        // Now we can use petgraph's depth_first_search directly!
        // The IntoNeighbors implementation will handle discovery automatically
        depth_first_search(&self.dynamic_graph, Some(start_node), |event| {
            match event {
                DfsEvent::Discover(node_idx, _) => {
                    if let Some(vertex) = self.dynamic_graph.get_vertex(node_idx) {
                        discovery_order.push(vertex.clone());
                        println!("DFS discovered: {:?}", vertex);
                    }
                    Control::Continue
                },
                DfsEvent::TreeEdge(from, to) => {
                    if let (Some(from_vertex), Some(to_vertex)) =
                        (self.dynamic_graph.get_vertex(from), self.dynamic_graph.get_vertex(to)) {
                        println!("DFS tree edge: {:?} -> {:?}", from_vertex, to_vertex);
                    }
                    Control::Continue
                },
                DfsEvent::BackEdge(from, to) => {
                    if let (Some(from_vertex), Some(to_vertex)) =
                        (self.dynamic_graph.get_vertex(from), self.dynamic_graph.get_vertex(to)) {
                        println!("DFS back edge: {:?} -> {:?}", from_vertex, to_vertex);
                    }
                    Control::Continue
                },
                _ => Control::Continue,
            }
        });

        discovery_order
    }

    /// Alternative: Use petgraph's Dfs iterator directly
    pub fn dfs_discover_iterator(&mut self, start: T) -> Vec<T> {
        let start_node = match self.dynamic_graph.add_start_vertex(start) {
            Some(node) => node,
            None => return Vec::new(),
        };

        let mut discovery_order = Vec::new();
        let mut dfs = petgraph::visit::Dfs::new(&self.dynamic_graph, start_node);

        while let Some(node_idx) = dfs.next(&self.dynamic_graph) {
            if let Some(vertex) = self.dynamic_graph.get_vertex(node_idx) {
                discovery_order.push(vertex.clone());
            }
        }

        discovery_order
    }

    /// Get the discovered graph
    pub fn get_graph(&self) -> &StableDiGraph<T, ()> {
        self.dynamic_graph.get_graph()
    }

    /// Get discovered vertices count
    pub fn discovered_count(&self) -> usize {
        self.dynamic_graph.discovered_count()
    }

    /// Get total edges count
    pub fn edges_count(&self) -> usize {
        self.dynamic_graph.edges_count()
    }
}

// Example implementation of GraphProvider for a simple numeric graph
pub struct SimpleGraphProvider {
    max_depth: usize,
    branching_factor: usize,
}

impl SimpleGraphProvider {
    pub fn new(max_depth: usize, branching_factor: usize) -> Self {
        Self { max_depth, branching_factor }
    }
}

impl GraphProvider<i32> for SimpleGraphProvider {
    fn get_neighbors(&mut self, vertex: &i32) -> Vec<i32> {
        if *vertex >= (self.max_depth as i32).pow(self.branching_factor as u32) {
            return Vec::new();
        }

        (1..=self.branching_factor)
            .map(|i| vertex * self.branching_factor as i32 + i)
            .collect()
    }

    fn vertex_exists(&mut self, vertex: &i32) -> bool {
        *vertex >= 0 && *vertex <= 1000
    }
}

// Example with external data source
pub struct ExternalDataProvider {
    call_count: usize,
}

impl ExternalDataProvider {
    pub fn new() -> Self {
        Self { call_count: 0 }
    }

    fn fetch_neighbors(&mut self, vertex: &String) -> Vec<String> {
        self.call_count += 1;
        println!("API call #{}: fetching neighbors for '{}'", self.call_count, vertex);

        match vertex.as_str() {
            "root" => vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "A" => vec!["A1".to_string(), "A2".to_string()],
            "B" => vec!["B1".to_string(), "B2".to_string(), "B3".to_string()],
            "C" => vec!["C1".to_string()],
            "A1" => vec!["A1a".to_string()],
            "A2" => vec!["A2a".to_string(), "A2b".to_string()],
            _ => Vec::new(),
        }
    }
}

impl GraphProvider<String> for ExternalDataProvider {
    fn get_neighbors(&mut self, vertex: &String) -> Vec<String> {
        std::thread::sleep(std::time::Duration::from_millis(10));
        self.fetch_neighbors(vertex)
    }

    fn vertex_exists(&mut self, vertex: &String) -> bool {
        !vertex.is_empty() && vertex.len() <= 10
    }
}

fn main() {
    println!("=== Dynamic Graph Discovery with IntoNeighbors ===");

    println!("\n--- Simple Numeric Graph ---");
    let provider = SimpleGraphProvider::new(3, 2);
    let mut discoverer = GraphDiscoverer::new(provider);
    let discovery_order = discoverer.dfs_discover(1);
    println!("Discovery order: {:?}", discovery_order);
    println!("Discovered {} vertices with {} edges",
             discoverer.discovered_count(),
             discoverer.edges_count());

    println!("\n--- External Data Source ---");
    let provider = ExternalDataProvider::new();
    let mut discoverer = GraphDiscoverer::new(provider);
    let discovery_order = discoverer.dfs_discover("root".to_string());
    println!("Discovery order: {:?}", discovery_order);
    println!("Discovered {} vertices with {} edges",
             discoverer.discovered_count(),
             discoverer.edges_count());

    println!("\n--- Using DFS Iterator ---");
    let provider = ExternalDataProvider::new();
    let mut discoverer = GraphDiscoverer::new(provider);
    let discovery_order = discoverer.dfs_discover_iterator("root".to_string());
    println!("Discovery order (iterator): {:?}", discovery_order);

    // Print final graph structure
    println!("\nFinal graph structure:");
    let graph = discoverer.get_graph();
    for node_idx in graph.node_indices() {
        if let Some(vertex) = graph.node_weight(node_idx) {
            let neighbors: Vec<_> = graph.neighbors(node_idx)
                .filter_map(|n| graph.node_weight(n))
                .collect();
            println!("  {:?} -> {:?}", vertex, neighbors);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_discovery() {
        let provider = SimpleGraphProvider::new(2, 2);
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover(1);
        assert!(!result.is_empty());
        assert!(result.contains(&1));
        assert!(discoverer.discovered_count() > 0);
    }

    #[test]
    fn test_external_provider_discovery() {
        let provider = ExternalDataProvider::new();
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover("root".to_string());
        assert!(result.contains(&"root".to_string()));
        assert!(result.len() > 1);
    }

    #[test]
    fn test_dfs_iterator() {
        let provider = SimpleGraphProvider::new(2, 2);
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover_iterator(1);
        assert!(!result.is_empty());
        assert!(result.contains(&1));
    }
}
