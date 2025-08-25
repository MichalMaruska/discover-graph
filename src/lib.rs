use petgraph::graph::{NodeIndex};
use petgraph::visit::{depth_first_search, DfsEvent, Control};
use petgraph::stable_graph::{StableDiGraph};

use std::collections::{HashMap};
use std::hash::Hash;

use tracing::debug;

/// Trait for providing vertices and edges on demand
pub trait GraphProvider<T> {
    /// Get neighbors of a given vertex
    fn get_neighbors(&mut self, vertex: &T) -> Vec<T>;

    /// Check if a vertex exists (optional validation)
    fn vertex_exists(&mut self, _vertex: &T) -> bool {
        true // Default implementation assumes all vertices exist
    }
}

mod dynamic_bit_set;

mod dynamic_graph;
use dynamic_graph::DynamicGraph;



/// Main discoverer that uses the dynamic graph wrapper
/// to invoke  `depth_first_search()'
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
                    Control::<()>::Continue // too bad not default for that B type.
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


    pub fn get_provider(self) -> (P, HashMap<T, NodeIndex>) {
        // Ref<'_, T>
        return (
            self.dynamic_graph.provider.into_inner(),
            self.dynamic_graph.vertex_to_node.into_inner()
        )
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
