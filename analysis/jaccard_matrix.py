# this generates a jaccard distance matrix of u

def loadgraph(path):
    """This takes a path parameter to a TSV
    The TSV is loaded, then parsed. The graph is stored as [{col1, col2}, {col1, col2}...]"""
    graph = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Store as a frozenset so order doesn't matter and can be used in sets
                graph.append(frozenset([parts[0], parts[1]]))
    return set(graph)

def jaccard(graphA, graphB):
    """
    graphA: a set of edges [{node1, node2}]
    graphB: a set of edges [{node1, node2}]
    
    This method finds the jaccard index of the edges graphA and graphB, ignoring order and returns it.
    """
    if len(graphA) == 0 and len(graphB) == 0:
        return 1.0
    intersection = len(graphA & graphB)
    union = len(graphA | graphB)
    return intersection / union if union > 0 else 0.0

def generateInterMatrix(graph_set_a):
    n = len(graph_set_a)
    data = [[0.0 for _ in range(n)] for _ in range(n)]
    for ind, a in enumerate(graph_set_a):
        for ind2, b in enumerate(graph_set_a):
            data[ind][ind2] = jaccard(a, b)
    
    return data

def generateIntraMatrix(graph_set_a, graph_set_b):
    """
    graph_set_a: A set of graphs [[{node1, node2}], [{node1, node2}]]
    graph_set_b: A different set of graphs
    """
    n = len(graph_set_a)
    m = len(graph_set_b)
    data = [[0.0 for _ in range(m)] for _ in range(n)]
    for ind, a in enumerate(graph_set_a):
        for ind2, b in enumerate(graph_set_b):
            data[ind][ind2] = jaccard(a, b)
    
    return data

# a = [loadgraph("data/F1.tsv"), loadgraph("data/F2.tsv"), loadgraph("data/F3.tsv")]
#b = [loadgraph("data/U1.tsv")]

#print(generateInterMatrix(a))
# print(generateIntraMatrix(a, b))


a = loadgraph("data/U2.tsv")
b = loadgraph("data/U1.tsv")

print(jaccard(a, b))

