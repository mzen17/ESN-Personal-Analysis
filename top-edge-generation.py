
from rag.graph import loadgraph, entity_collapse, query
from pathlib import Path

data = ["U1.tsv", "U2.tsv", "U3.tsv", "U4.tsv", 
        "F1.tsv", "F2.tsv", "F3.tsv", "F4.tsv"]

# Load all graphs
graphs = []
for filename in data:
    path = Path("data") / filename
    graphs.append(loadgraph(str(path)))

# Create union of all graphs
union_graph = []
for graph in graphs:
    union_graph.extend(graph)

print(f"Union graph has {len(union_graph)} edges")

# Collapse entities on the union
collapsed_rag = entity_collapse(union_graph, clustering_tr=0.5)
print(f"Collapsed to {len(collapsed_rag)} unique entities")

# Collapse each individual graph for querying
collapsed_graphs = []
for i, graph in enumerate(graphs):
    print(f"Collapsing graph {i+1}/8...")
    collapsed_graphs.append(entity_collapse(graph, clustering_tr=0.5))

# For each entity in collapsed union, query each graph
# Store the count of graphs with semantic similarity > 0.2
T = []

for entity_label, _ in collapsed_rag:
    matching_graphs = []
    
    # Query each of the 8 collapsed graphs
    for i, collapsed_graph in enumerate(collapsed_graphs):
        result = query(collapsed_graph, entity_label, return_similarity=True)
        
        if result:
            matched_label, images, similarity = result
            if similarity > 0.3:
                matching_graphs.append((data[i].replace('.tsv', ''), similarity))
    
    T.append((entity_label, len(matching_graphs), matching_graphs))

# Sort by count (descending) then by label
T.sort(key=lambda x: (-x[1], x[0]))

# Print the results
print("\n" + "="*80)
print("ENTITY PREVALENCE ACROSS GRAPHS (similarity > 0.2)")
print("="*80)
print(f"{'Entity':<40} {'Count':>6} {'Matching Graphs'}")
print("-"*80)

for entity, count, matching_graphs in T:
    # Format matching graphs with their similarity scores
    graph_str = ", ".join([f"{g}({s:.2f})" for g, s in matching_graphs])
    print(f"{entity:<40} {count:>6}  {graph_str}")

