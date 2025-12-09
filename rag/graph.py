from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Use sentence transformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')


def loadgraph(path):
    """This takes a path parameter to a TSV
    The TSV is loaded, then parsed. The graph is stored as [(img1, img2, edge_label), ...]"""
    graph = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                # Store as tuple: (img1, img2, edge_label)
                graph.append((parts[0], parts[1], parts[3]))
    return graph

def entity_collapse(graph, clustering_tr = 0.4):
    """graph is a set of edges of [img1, img2, edge]
    What this function does is that it extracts the set of edges. Then, it uses an unsupervised clustering technique to merge the distinct edge patterns singular list of distinct edge weights (mounds are labeled by edge weights (string)). Finally, it creates a new parameter graph of [edge weight, [img1, img2, img3, ...]]."""
        
    # First, collect edge labels and their associated images
    edge_label_to_images = {}
    for img1, img2, edge_label in graph:
        if edge_label not in edge_label_to_images:
            edge_label_to_images[edge_label] = set()
        edge_label_to_images[edge_label].add(img1)
        edge_label_to_images[edge_label].add(img2)
    
    # Get unique edge labels
    unique_labels = list(edge_label_to_images.keys())
    
    if len(unique_labels) == 0:
        return []
    
    if len(unique_labels) == 1:
        return [(unique_labels[0], sorted(list(edge_label_to_images[unique_labels[0]])))]
    
    # Use sentence transformer for semantic similarity
    embeddings = model.encode(unique_labels)
    
    # Perform agglomerative clustering
    # Use distance threshold to determine number of clusters automatically
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_tr, metric='cosine', linkage='average')
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Group edge labels by cluster
    clusters = {}
    for label, cluster_id in zip(unique_labels, cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = {'labels': [], 'images': set()}
        clusters[cluster_id]['labels'].append(label)
        clusters[cluster_id]['images'].update(edge_label_to_images[label])
    
    # Create result with representative label (most common or first) for each cluster
    result = []
    for cluster_id, cluster_data in clusters.items():
        # Use the first label as representative (could use most frequent instead)
        representative_label = cluster_data['labels'][0]
        if len(cluster_data['labels']) > 1:
            representative_label = f"{representative_label}"
        result.append((representative_label, sorted(list(cluster_data['images']))))
    
    return result

def query(graph, term, return_similarity=False):
    """Query the graph for a term and return the best matching pair.
    graph is a list of [(label, [images])], term is a string to search for.
    Returns the (label, images) pair with highest cosine similarity to the term.
    If return_similarity=True, returns (label, images, similarity_score)."""
    
    if not graph:
        return None
    
    # Extract all labels from the graph
    labels = [label for label, images in graph]
    
    
    # Encode the query term and all labels
    term_embedding = model.encode([term])[0]
    label_embeddings = model.encode(labels)
    
    # Calculate cosine similarities
    similarities = cosine_similarity([term_embedding], label_embeddings)[0]
    
    # Find the index with highest similarity
    best_match_idx = np.argmax(similarities)
    
    # Return the matching pair
    if return_similarity:
        return graph[best_match_idx][0], graph[best_match_idx][1], similarities[best_match_idx]
    else:
        return graph[best_match_idx]
