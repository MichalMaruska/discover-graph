use discover_graph::GraphDiscoverer;

use discover_graph::SimpleGraphProvider;
use discover_graph::ExternalDataProvider;



fn demonstrate_graph_access() {
    println!("=== Understanding Graph Access Patterns ===");

    let provider = ExternalDataProvider::new();
    let mut discoverer = GraphDiscoverer::new(provider);

    // 1. Perform discovery
    let discovery_order = discoverer.dfs_discover("root".to_string()); // mmc: so here T = String
    println!("Discovered: {:?}", discovery_order);

    // 2. Access the "snapshot" StableDiGraph (no further discovery)
    let static_graph = discoverer.get_graph();
    println!("\n--- Graph Snapshot Analysis ---");
    println!("Nodes: {}, Edges: {}", static_graph.node_count(), static_graph.edge_count());

    // You can use any petgraph algorithm on this:
    for node_idx in static_graph.node_indices() {
        if let Some(vertex) = static_graph.node_weight(node_idx) {
            let neighbor_count = static_graph.neighbors(node_idx).count();
            println!("Vertex {:?} has {} neighbors (static view)", vertex, neighbor_count);
        }
    }

    // 3. Access the dynamic wrapper (can trigger more discovery)
    let dynamic_graph = discoverer.get_dynamic_graph();
    println!("\n--- Dynamic Graph Wrapper ---");
    println!("Discovery calls made: {}", dynamic_graph.discovered_count());

    // 4. The key difference:
    println!("\n--- Key Difference ---");
    println!("static_graph.neighbors() -> Uses existing edges only");
    println!("dynamic_graph (via IntoNeighbors) -> May discover new neighbors");
}


fn main() {
    demonstrate_graph_access();

    println!("\n=== Dynamic Graph Discovery with IntoNeighbors ===");

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
