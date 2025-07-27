use petgraph::graph::{NodeIndex,DefaultIx};
use petgraph::visit::{depth_first_search, DfsEvent, Control, IntoNeighbors, GraphBase, Visitable};
use petgraph::stable_graph::{StableDiGraph};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::cell::RefCell;
use tracing::debug;

mod dynamic_bit_set;
use dynamic_bit_set::DynamicBitSet;

/// Trait for providing vertices and edges on demand
pub trait GraphProvider<T> {
    /// Get neighbors of a given vertex
    fn get_neighbors(&mut self, vertex: &T) -> Vec<T>;

    /// Check if a vertex exists (optional validation)
    fn vertex_exists(&mut self, _vertex: &T) -> bool {
        true // Default implementation assumes all vertices exist
    }
}

/// Dynamic graph wrapper that implements IntoNeighbors with on-demand discovery
pub struct DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    graph: RefCell<StableDiGraph<T, ()>>,
    // mmc: why T and not &T ? cannot we share it with graph?
    vertex_to_node: RefCell<HashMap<T, NodeIndex>>,
    // transforming external nodes to Graph vertices.
    // Graph provides the reverse. But why 2 copies of T?

    discovered: RefCell<HashSet<T>>, // why?   discover_if_needed...have we invoked provider?
    provider: RefCell<P>,
}

impl<T, P> DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    pub fn new(provider: P) -> Self {
        Self {
            graph: RefCell::new(StableDiGraph::new()),
            vertex_to_node: RefCell::new(HashMap::new()),
            discovered: RefCell::new(HashSet::new()),
            provider: RefCell::new(provider),
        }
    }

    /// Get or create a node in the graph for the given vertex
    fn get_or_create_node(&self, vertex: T) -> NodeIndex {
        let mut vertex_to_node = self.vertex_to_node.borrow_mut();
        if let Some(&node_idx) = vertex_to_node.get(&vertex) {
            node_idx
        } else {
            let mut graph = self.graph.borrow_mut();
            let node_idx = graph.add_node(vertex.clone()); // so 2 copies!
            vertex_to_node.insert(vertex, node_idx);
            node_idx
        }
    }

    /// Discover neighbors for a vertex and add them to the graph
    /// This is called by the IntoNeighbors implementation
    fn discover_if_needed(&self, node_idx: NodeIndex) {
        // Get the vertex from the node
        let vertex = {
            let graph = self.graph.borrow();
            graph[node_idx].clone()
        };

        // Check if already discovered
        {
            let discovered = self.discovered.borrow();
            if discovered.contains(&vertex) {
                return;
            }
        }

        // Get neighbors from provider
        let neighbors = self.provider.borrow_mut().get_neighbors(&vertex);

        // Mark as discovered
        self.discovered.borrow_mut().insert(vertex.clone());

        // eprintln!("Dynamically discovering vertex {:?} with {} neighbors", vertex, neighbors.len());

        // Add neighbors to graph
        for neighbor in neighbors {
            if self.provider.borrow_mut().vertex_exists(&neighbor) {
                // external node -> graph node
                let neighbor_node = self.get_or_create_node(neighbor.clone());

                // Add edge if it doesn't exist
                let mut graph = self.graph.borrow_mut();
                if graph.find_edge(node_idx, neighbor_node).is_none() {
                    // eprintln!("Dynamically adding edge");
                    graph.add_edge(node_idx, neighbor_node, ());
                }
            }
        }
    }

    pub fn add_start_vertex(&self, vertex: T) -> Option<NodeIndex> {
        if self.provider.borrow_mut().vertex_exists(&vertex) {
            Some(self.get_or_create_node(vertex))
        } else {
            None
        }
    }

    pub fn get_vertex(&self, node_idx: NodeIndex) -> Option<T> {
        let graph = self.graph.borrow();
        graph.node_weight(node_idx).cloned()
    }

    pub fn get_graph_snapshot(&self) -> StableDiGraph<T, ()> {
        self.graph.borrow().clone()
    }

    pub fn discovered_count(&self) -> usize {
        self.discovered.borrow().len()
    }

    pub fn edges_count(&self) -> usize {
        self.graph.borrow().edge_count()
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

// mmc: IntoNeighbors + Visitable
// Implement Visitable for our wrapper
/*
impl<T, P> Visitable for DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    // must by dynamic too!
    type Map = <StableDiGraph<T, ()> as Visitable>::Map;

    // fixme: I need to grow it on demand!
    fn visit_map(&self) -> Self::Map {
        self.graph.borrow().visit_map()
            // mmc: I need dynamic one!
    }

    fn reset_map(&self, map: &mut Self::Map) {
        self.graph.borrow().reset_map(map)
    }
}
*/
// Implement Visitable for our wrapper with dynamic bit set
impl<T, P> Visitable for DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    type Map = DynamicBitSet;

    fn visit_map(&self) -> Self::Map {
        // Create a bit set with current graph capacity
        DynamicBitSet::with_capacity(self.graph.borrow().node_count())
    }

    fn reset_map(&self, map: &mut Self::Map) {
        map.reset();
    }
}



// Custom neighbors iterator that triggers discovery
pub struct DynamicNeighbors<'a, T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    graph: &'a DynamicGraph<T, P>,
    inner: petgraph::stable_graph::WalkNeighbors<DefaultIx>, // petgraph::Directed  Neighbors
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

            //
            self.inner = self.graph.graph.borrow().neighbors(self.node_idx).detach();
        }

        self.inner.next_node(&*self.graph.graph.borrow()) // wtf!
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
            inner: self.graph.borrow().neighbors(node_idx).detach(),
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
            // inside the graph.
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
                        debug!("DFS discovered: {:?}", vertex);
                    }
                    Control::<()>::Continue
                },
                DfsEvent::TreeEdge(from, to) => {
                    if let (Some(from_vertex), Some(to_vertex)) =
                        (self.dynamic_graph.get_vertex(from), self.dynamic_graph.get_vertex(to)) {
                            debug!("DFS tree edge: {:?} -> {:?}", from_vertex, to_vertex);
                    }
                    Control::Continue
                },
                DfsEvent::BackEdge(from, to) => {
                    if let (Some(from_vertex), Some(to_vertex)) =
                        (self.dynamic_graph.get_vertex(from), self.dynamic_graph.get_vertex(to)) {
                        debug!("DFS back edge: {:?} -> {:?}", from_vertex, to_vertex);
                    }
                    Control::Continue
                },
                DfsEvent::Finish(node_idx, _) => {
                    if let Some(vertex) = self.dynamic_graph.get_vertex(node_idx) {
                        debug!("finished with {:?}", vertex);
                        discovery_order.push(vertex.clone());
                    }
                    // self.dynamic_graph.graph.borrow().node_weight(node_idx));
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
                discovery_order.push(vertex);
            }
        }

        discovery_order
    }

    /// Get the discovered graph - returns immutable reference to the underlying StableDiGraph
    ///
    /// This provides access to the petgraph data structure after discovery is complete.
    /// You can use this for:
    /// - Analyzing graph structure (node_count, edge_count, etc.)
    /// - Running other petgraph algorithms on the discovered graph
    /// - Accessing node weights and edge data
    /// - Manual traversal using petgraph's iterators
    ///
    /// Note: This returns a snapshot of the StableDiGraph, not the DynamicGraph wrapper,
    /// so it won't trigger further discovery if you traverse it.
    pub fn get_graph(&self) -> StableDiGraph<T, ()> {
        self.dynamic_graph.get_graph_snapshot()
    }

    /// Get access to the dynamic wrapper (for further discovery operations)
    pub fn get_dynamic_graph(&self) -> &DynamicGraph<T, P> {
        &self.dynamic_graph
    }

    /// Get a mutable reference to the dynamic wrapper
    pub fn get_dynamic_graph_mut(&mut self) -> &mut DynamicGraph<T, P> {
        &mut self.dynamic_graph
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


// mod simple {
// fixme: for testing only!
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
            .map(|i| vertex * self.branching_factor as i32 + i as i32)
            .collect()
    }

    fn vertex_exists(&mut self, vertex: &i32) -> bool {
        *vertex >= 0 && *vertex <= 1000
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


    /*
    #[ignore]
    #[test]
    fn test_external_provider_discovery() {
        let provider = ExternalDataProvider::new();
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover("root".to_string());
        assert!(result.contains(&"root".to_string()));
        assert!(result.len() > 1);
    }
    */

    #[test]
    fn test_dfs_iterator() {
        let provider = SimpleGraphProvider::new(2, 2);
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover_iterator(1);
        assert!(!result.is_empty());
        assert!(result.contains(&1));
    }
}
