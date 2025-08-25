use petgraph::graph::{NodeIndex,DefaultIx};
use petgraph::visit::{IntoNeighbors, GraphBase, Visitable};
use petgraph::stable_graph::{StableDiGraph};

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::cell::RefCell;

use tracing::debug;

// ::crate::    global paths cannot start with `crate`
use crate::dynamic_bit_set::DynamicBitSet;
use crate::GraphProvider;


/// Dynamic graph wrapper that implements IntoNeighbors with on-demand discovery
pub struct DynamicGraph<T, P>
where
    T: Clone + Eq + Hash + std::fmt::Debug,
    P: GraphProvider<T>,
{
    graph: RefCell<StableDiGraph<T, ()>>,
    // mmc: why T and not &T ? cannot we share it with graph?
    pub(crate) vertex_to_node: RefCell<HashMap<T, NodeIndex>>,
    // transforming external nodes to Graph vertices.
    // Graph provides the reverse. But why 2 copies of T?

    discovered: RefCell<HashSet<T>>, // why?   discover_if_needed...have we invoked provider?
    pub(crate) provider: RefCell<P>,
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

        debug!("Dynamically discovering vertex {:?} with {} neighbors", vertex, neighbors.len());

        // Add neighbors to graph
        for neighbor in neighbors {
            if self.provider.borrow_mut().vertex_exists(&neighbor) {
                // external node -> graph node
                let neighbor_node = self.get_or_create_node(neighbor.clone());

                // Add edge if it doesn't exist
                let mut graph = self.graph.borrow_mut();
                if graph.find_edge(node_idx, neighbor_node).is_none() {
                    debug!("Dynamically adding edge");
                    graph.add_edge(node_idx, neighbor_node, ()); // could add a name/weight
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
