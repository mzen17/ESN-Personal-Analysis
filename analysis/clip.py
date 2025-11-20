from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from collections import defaultdict
import argparse
import json

def load_images_from_directory(imagedir):
    """Recursively load all image files from a directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(imagedir).rglob(f'*{ext}'))
        image_paths.extend(Path(imagedir).rglob(f'*{ext.upper()}'))
    
    return sorted([str(p) for p in image_paths])

def compute_embeddings(model, image_paths):
    """Compute CLIP embeddings for all images."""
    embeddings = []
    valid_paths = []
    
    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path)
            embedding = model.encode(image)
            embeddings.append(embedding)
            valid_paths.append(img_path)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return np.array(embeddings), valid_paths

def cluster_images(embeddings, eps=0.5, min_samples=2):
    """Cluster images using DBSCAN based on cosine similarity.
    
    Args:
        embeddings: numpy array of image embeddings
        eps: The maximum distance between two samples for one to be considered
             as in the neighborhood of the other (similarity threshold: 1-eps)
        min_samples: The number of samples in a neighborhood for a point to be
                    considered as a core point
    """
    # Use cosine distance (1 - cosine_similarity) for clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    
    return labels

def compute_2d_projection(embeddings, method='tsne', random_state=42):
    """Compute 2D projection of embeddings for visualization.
    
    Args:
        embeddings: numpy array of image embeddings
        method: 'tsne' or 'umap' for dimensionality reduction
        random_state: random seed for reproducibility
    
    Returns:
        2D numpy array of shape (n_samples, 2)
    """
    print(f"\nComputing 2D projection using {method.upper()}...")
    
    if method.lower() == 'tsne':
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, len(embeddings) - 1),
            max_iter=1000,
            metric='cosine'
        )
        coords_2d = tsne.fit_transform(embeddings)
    elif method.lower() == 'umap':
        try:
            from umap import UMAP
            umap_model = UMAP(
                n_components=2,
                random_state=random_state,
                metric='cosine',
                n_neighbors=min(15, len(embeddings) - 1)
            )
            coords_2d = umap_model.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            return compute_2d_projection(embeddings, method='tsne', random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")
    
    return coords_2d

def organize_clusters(image_paths, labels):
    """Organize images into piles/clusters."""
    clusters = defaultdict(list)
    
    for img_path, label in zip(image_paths, labels):
        if label == -1:
            clusters['noise'].append(img_path)
        else:
            clusters[f'pile_{label}'].append(img_path)
    
    return dict(clusters)

def create_visualization_data(image_paths, embeddings, labels, coords_2d):
    """Create a comprehensive data structure for visualization.
    
    Returns a dictionary with:
    - images: list of image data with paths, coordinates, and cluster labels
    - clusters: cluster summary information
    - metadata: overall statistics
    """
    # Normalize coordinates to [0, 1] range for easier visualization
    coords_normalized = coords_2d.copy()
    coords_normalized[:, 0] = (coords_normalized[:, 0] - coords_normalized[:, 0].min()) / \
                               (coords_normalized[:, 0].max() - coords_normalized[:, 0].min())
    coords_normalized[:, 1] = (coords_normalized[:, 1] - coords_normalized[:, 1].min()) / \
                               (coords_normalized[:, 1].max() - coords_normalized[:, 1].min())
    
    # Create image data
    images_data = []
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        images_data.append({
            'path': img_path,
            'filename': Path(img_path).name,
            'cluster': int(label),
            'cluster_name': 'noise' if label == -1 else f'pile_{label}',
            'coordinates': {
                'x': float(coords_normalized[i, 0]),
                'y': float(coords_normalized[i, 1])
            },
            'coordinates_raw': {
                'x': float(coords_2d[i, 0]),
                'y': float(coords_2d[i, 1])
            }
        })
    
    # Create cluster summaries
    clusters_summary = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_coords = coords_normalized[cluster_indices]
        
        # Calculate cluster statistics
        if len(cluster_embeddings) > 1:
            similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    sim = util.cos_sim(cluster_embeddings[i], cluster_embeddings[j]).item()
                    similarities.append(sim)
            avg_similarity = float(np.mean(similarities))
            std_similarity = float(np.std(similarities))
        else:
            avg_similarity = 1.0
            std_similarity = 0.0
        
        cluster_name = 'noise' if label == -1 else f'pile_{label}'
        clusters_summary[cluster_name] = {
            'label': int(label),
            'size': len(cluster_indices),
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'centroid': {
                'x': float(np.mean(cluster_coords[:, 0])),
                'y': float(np.mean(cluster_coords[:, 1]))
            },
            'members': [image_paths[i] for i in cluster_indices]
        }
    
    # Create metadata
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metadata = {
        'total_images': len(image_paths),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'embedding_dim': embeddings.shape[1],
        'visualization_method': 'tsne',
        'coordinate_range': 'normalized [0, 1]'
    }
    
    return {
        'images': images_data,
        'clusters': clusters_summary,
        'metadata': metadata
    }

def print_cluster_stats(clusters, embeddings, labels):
    """Print statistics about the clusters."""
    print("\n" + "="*80)
    print("CLUSTERING RESULTS")
    print("="*80)
    
    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nTotal images processed: {len(labels)}")
    print(f"Number of piles/clusters: {n_clusters}")
    print(f"Images in noise (outliers): {n_noise}")
    
    # Print details for each cluster
    for cluster_name in sorted(clusters.keys()):
        if cluster_name == 'noise':
            continue
        
        images = clusters[cluster_name]
        cluster_id = int(cluster_name.split('_')[1])
        
        # Get embeddings for this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate average pairwise similarity within cluster
        if len(cluster_embeddings) > 1:
            similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    sim = util.cos_sim(cluster_embeddings[i], cluster_embeddings[j]).item()
                    similarities.append(sim)
            avg_similarity = np.mean(similarities)
        else:
            avg_similarity = 1.0
        
        print(f"\n{cluster_name}:")
        print(f"  Size: {len(images)}")
        print(f"  Avg similarity: {avg_similarity:.4f}")
        print(f"  Images:")
        for img in images:
            print(f"    - {Path(img).name}")
    
    if 'noise' in clusters:
        print(f"\nNoise (outliers):")
        print(f"  Size: {len(clusters['noise'])}")
        print(f"  Images:")
        for img in clusters['noise']:
            print(f"    - {Path(img).name}")

def main():
    parser = argparse.ArgumentParser(
        description='Cluster images based on CLIP embeddings and create 2D visualization'
    )
    parser.add_argument(
        '--imagedir',
        type=str,
        default='workdata',
        help='Directory containing images to cluster'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='DBSCAN eps parameter (lower = stricter clustering, try 0.2-0.5)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help='DBSCAN min_samples parameter (minimum cluster size)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='image_clusters_visualization.json',
        help='Output JSON file for cluster results with 2D coordinates'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap'],
        help='Dimensionality reduction method for 2D visualization'
    )
    
    args = parser.parse_args()
    
    print("Loading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")
    
    print(f"\nScanning directory: {args.imagedir}")
    image_paths = load_images_from_directory(args.imagedir)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Exiting.")
        return
    
    print("\nComputing embeddings...")
    embeddings, valid_paths = compute_embeddings(model, image_paths)
    print(f"Successfully processed {len(valid_paths)} images")
    
    if len(valid_paths) < 2:
        print("Need at least 2 valid images to cluster! Exiting.")
        return
    
    print(f"\nClustering with eps={args.eps}, min_samples={args.min_samples}...")
    labels = cluster_images(embeddings, eps=args.eps, min_samples=args.min_samples)
    
    # Compute 2D projection for visualization
    coords_2d = compute_2d_projection(embeddings, method=args.method)
    
    clusters = organize_clusters(valid_paths, labels)
    
    print_cluster_stats(clusters, embeddings, labels)
    
    # Create comprehensive visualization data
    print("\nCreating visualization data...")
    viz_data = create_visualization_data(valid_paths, embeddings, labels, coords_2d)
    
    # Save results to JSON
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("\nVisualization data structure:")
    print(f"  - {len(viz_data['images'])} images with 2D coordinates")
    print(f"  - {len(viz_data['clusters'])} clusters")
    print(f"  - Coordinates normalized to [0, 1] range")
    print("\nDone!")

if __name__ == "__main__":
    main()
