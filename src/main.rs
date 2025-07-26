use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::{depth_first_search, DfsEvent, Control};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Trait for providing vertices and edges on demand
pub trait GraphProvider<T> {
    /// Get neighbors of a given vertex
    fn get_neighbors(&mut self, vertex: &T) -> Vec<T>;

    /// Check if a vertex exists (optional validation)
    fn vertex_exists(&mut self, vertex: &T) -> bool {
        true // Default implementation assumes all vertices exist
    }
}

/// On-demand graph discoverer using petgraph's depth_first_search
pub struct GraphDiscoverer<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    provider: P,
    graph: DiGraph<T, ()>,
    vertex_to_node: HashMap<T, NodeIndex>,
    discovered: HashSet<T>,
    discovery_order: Vec<T>,
    pending_discovery: Vec<T>,
}

impl<T, P> GraphDiscoverer<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            graph: DiGraph::new(),
            vertex_to_node: HashMap::new(),
            discovered: HashSet::new(),
            discovery_order: Vec::new(),
            pending_discovery: Vec::new(),
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
    fn discover_vertex(&mut self, vertex: &T) -> Vec<NodeIndex> {
        if self.discovered.contains(vertex) {
            // Already discovered, just return existing neighbors
            let node_idx = self.vertex_to_node[vertex];
            return self.graph.neighbors(node_idx).collect();
        }

        let neighbors = self.provider.get_neighbors(vertex);
        self.discovered.insert(vertex.clone());

        println!("Discovering vertex {:?} with {} neighbors", vertex, neighbors.len());

        let current_node = self.vertex_to_node[vertex];
        let mut neighbor_nodes = Vec::new();

        // Add neighbors to graph
        for neighbor in neighbors {
            if self.provider.vertex_exists(&neighbor) {
                let neighbor_node = self.get_or_create_node(neighbor.clone());

                // Add edge if it doesn't exist
                if self.graph.find_edge(current_node, neighbor_node).is_none() {
                    self.graph.add_edge(current_node, neighbor_node, ());
                }

                neighbor_nodes.push(neighbor_node);

                // Track for potential future discovery
                if !self.discovered.contains(&neighbor) {
                    self.pending_discovery.push(neighbor);
                }
            }
        }

        neighbor_nodes
    }

    /// Perform DFS with on-demand discovery using petgraph's depth_first_search
    pub fn dfs_discover(&mut self, start: T) -> Vec<T> {
        // Validate start vertex
        if !self.provider.vertex_exists(&start) {
            return Vec::new();
        }

        // Clear previous results
        self.discovery_order.clear();
        self.pending_discovery.clear();

        // Add start vertex to graph
        let start_node = self.get_or_create_node(start.clone());

        // Initial discovery of start vertex
        self.discover_vertex(&start);

        // Keep running DFS until no new vertices are discovered
        loop {
            let initial_node_count = self.graph.node_count();

            // Run petgraph's depth_first_search
            depth_first_search(&self.graph, Some(start_node), |event| {
                match event {
                    DfsEvent::Discover(node_idx, _) => {
                        let vertex = &self.graph[node_idx];
                        self.discovery_order.push(vertex.clone());

                        // Discover neighbors on-demand
                        self.discover_vertex(vertex);

                        Control::<()>::Continue
                    },
                    DfsEvent::TreeEdge(_, target) => {
                        // Ensure target vertex is discovered
                        let target_vertex = &self.graph[target];
                        self.discover_vertex(target_vertex);
                        Control::Continue
                    },
                    _ => Control::Continue,
                }
            });

            // If no new nodes were added, we're done
            if self.graph.node_count() == initial_node_count {
                break;
            }
        }

        self.discovery_order.clone()
    }

    /// Alternative approach: iterative DFS with on-demand discovery
    pub fn dfs_discover_iterative(&mut self, start: T) -> Vec<T> {
        if !self.provider.vertex_exists(&start) {
            return Vec::new();
        }

        self.discovery_order.clear();
        let start_node = self.get_or_create_node(start);

        // Continue until no more vertices can be discovered
        loop {
            let nodes_before = self.graph.node_count();

            // Use petgraph's DFS on current graph state
            depth_first_search(&self.graph, Some(start_node), |event| {
                if let DfsEvent::Discover(node_idx, _) = event {
                    let vertex = &self.graph[node_idx].clone();

                    // Only add to discovery order if not already added
                    if !self.discovery_order.contains(vertex) {
                        self.discovery_order.push(vertex.clone());
                    }

                    // Discover neighbors for this vertex
                    self.discover_vertex(vertex);
                }
                Control::<()>::Continue
            });

            // If no new nodes were discovered, we're done
            if self.graph.node_count() == nodes_before {
                break;
            }
        }

        self.discovery_order.clone()
    }

    /// Use petgraph's DFS with a visitor pattern for more control
    pub fn dfs_discover_with_visitor<F>(&mut self, start: T, mut visitor: F) -> Vec<T>
    where
        F: FnMut(&T, &[T]) -> bool, // Returns true to continue exploring
    {
        if !self.provider.vertex_exists(&start) {
            return Vec::new();
        }

        self.discovery_order.clear();
        let start_node = self.get_or_create_node(start);

        loop {
            let nodes_before = self.graph.node_count();

            depth_first_search(&self.graph, Some(start_node), |event| {
                match event {
                    DfsEvent::Discover(node_idx, _) => {
                        let vertex = &self.graph[node_idx].clone();

                        if !self.discovery_order.contains(vertex) {
                            self.discovery_order.push(vertex.clone());

                            // Get current neighbors
                            let current_neighbors: Vec<T> = self.graph
                                .neighbors(node_idx)
                                .map(|n| self.graph[n].clone())
                                .collect();

                            // Call visitor - if it returns false, stop exploring this branch
                            if !visitor(vertex, &current_neighbors) {
                                return Control::<()>::Prune;
                            }

                            // Discover new neighbors
                            self.discover_vertex(vertex);
                        }

                        Control::Continue
                    },
                    _ => Control::Continue,
                }
            });

            if self.graph.node_count() == nodes_before {
                break;
            }
        }

        self.discovery_order.clone()
    }

    /// Get the discovered graph
    pub fn get_graph(&self) -> &DiGraph<T, ()> {
        &self.graph
    }

    /// Get discovered vertices count
    pub fn discovered_count(&self) -> usize {
        self.discovered.len()
    }

    /// Get total edges count
    pub fn edges_count(&self) -> usize {
        self.graph.edge_count()
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
        // Simple rule: each vertex n has neighbors [n*branching_factor+1, n*branching_factor+2, ...]
        if *vertex >= (self.max_depth as i32).pow(self.branching_factor as u32) {
            return Vec::new(); // No neighbors for deep vertices
        }

        (1..=self.branching_factor)
            .map(|i| vertex * self.branching_factor as i32 + i)
            .collect()
    }

    fn vertex_exists(&mut self, vertex: &i32) -> bool {
        *vertex >= 0 && *vertex <= 1000 // Arbitrary limit
    }
}

// Example with external data source (simulated)
pub struct ExternalDataProvider {
    call_count: usize,
}

impl ExternalDataProvider {
    pub fn new() -> Self {
        Self { call_count: 0 }
    }

    // Simulate external API call
    fn fetch_neighbors(&mut self, vertex: &String) -> Vec<String> {
        self.call_count += 1;
        println!("API call #{}: fetching neighbors for '{}'", self.call_count, vertex);

        // Simulate different neighbor patterns based on vertex name
        match vertex.as_str() {
            "root" => vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "A" => vec!["A1".to_string(), "A2".to_string()],
            "B" => vec!["B1".to_string(), "B2".to_string(), "B3".to_string()],
            "C" => vec!["C1".to_string()],
            "A1" => vec!["A1a".to_string()],
            "A2" => vec!["A2a".to_string(), "A2b".to_string()],
            _ => Vec::new(), // Leaf nodes
        }
    }
}

impl GraphProvider<String> for ExternalDataProvider {
    fn get_neighbors(&mut self, vertex: &String) -> Vec<String> {
        // Simulate network delay
        std::thread::sleep(std::time::Duration::from_millis(10));
        self.fetch_neighbors(vertex)
    }

    fn vertex_exists(&mut self, vertex: &String) -> bool {
        !vertex.is_empty() && vertex.len() <= 10
    }
}

fn main() {
    println!("=== Example 1: Simple Numeric Graph with petgraph DFS ===");
    let provider = SimpleGraphProvider::new(3, 2);
    let mut discoverer = GraphDiscoverer::new(provider);

    let discovery_order = discoverer.dfs_discover(1);
    println!("Discovery order: {:?}", discovery_order);
    println!("Discovered {} vertices with {} edges",
             discoverer.discovered_count(),
             discoverer.edges_count());

    println!("\n=== Example 2: External Data Source with Iterative DFS ===");
    let provider = ExternalDataProvider::new();
    let mut discoverer = GraphDiscoverer::new(provider);

    let discovery_order = discoverer.dfs_discover_iterative("root".to_string());
    println!("Discovery order: {:?}", discovery_order);
    println!("Discovered {} vertices with {} edges",
             discoverer.discovered_count(),
             discoverer.edges_count());

    println!("\n=== Example 3: DFS with Custom Visitor ===");
    let provider = ExternalDataProvider::new();
    let mut discoverer = GraphDiscoverer::new(provider);

    let discovery_order = discoverer.dfs_discover_with_visitor("root".to_string(), |vertex, neighbors| {
        println!("Visiting: {} (current neighbors: {:?})", vertex, neighbors);
        // Stop exploring branches that start with "B"
        !vertex.starts_with("B")
    });
    println!("Discovery order with visitor: {:?}", discovery_order);

    // Print the final graph structure
    println!("\nFinal graph structure:");
    let graph = discoverer.get_graph();
    for node_idx in graph.node_indices() {
        let vertex = &graph[node_idx];
        let neighbors: Vec<_> = graph.neighbors(node_idx)
            .map(|n| &graph[n])
            .collect();
        println!("  {:?} -> {:?}", vertex, neighbors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_petgraph_dfs_discovery() {
        let provider = SimpleGraphProvider::new(2, 2);
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover(1);
        assert!(!result.is_empty());
        assert!(result.contains(&1));
    }

    #[test]
    fn test_iterative_dfs() {
        let provider = ExternalDataProvider::new();
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover_iterative("root".to_string());
        assert!(result.contains(&"root".to_string()));
        assert!(result.len() > 1);
    }

    #[test]
    fn test_visitor_pattern() {
        let provider = ExternalDataProvider::new();
        let mut discoverer = GraphDiscoverer::new(provider);

        let result = discoverer.dfs_discover_with_visitor("root".to_string(), |_, _| true);
        assert!(result.contains(&"root".to_string()));
    }
}
