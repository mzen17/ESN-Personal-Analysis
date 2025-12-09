# baseline script
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from pathlib import Path
import json

def generate_graph():
    """
    Compute CLIP similarity scores between all pairs of images in workdata.
    Returns a dict mapping each image to a list of images with similarity > 0.8.
    """
    # Load CLIP model
    print("Loading CLIP model...")
    model = SentenceTransformer("clip-ViT-B-32")
    
    # Get all images from workdata
    workdata_path = Path("workdata")
    image_paths = sorted(list(workdata_path.glob("*.jpg")) + list(workdata_path.glob("*.png")))
    
    if len(image_paths) != 30:
        print(f"Warning: Found {len(image_paths)} images, expected 30")
    
    print(f"\nFound {len(image_paths)} images")
    print("Computing embeddings...")
    
    # Compute embeddings for all images
    embeddings = []
    valid_paths = []
    
    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path)
            embedding = model.encode(image)
            embeddings.append(embedding)
            valid_paths.append(str(img_path))
            print(f"  Processed {i + 1}/{len(image_paths)}: {img_path.name}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    print(f"\nComputing pairwise similarity scores...")
    
    # Build similarity graph: dict of image -> list of similar images
    similarity_graph = {}
    
    for i, img1_path in enumerate(valid_paths):
        similar_images = []
        
        for j, img2_path in enumerate(valid_paths):
            if i != j:  # Don't compare image with itself
                # Compute cosine similarity
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                
                if similarity > 0.85:
                    similar_images.append(img2_path)
        
        similarity_graph[img1_path] = similar_images
        print(f"  {Path(img1_path).name}: {len(similar_images)} similar images")
    
    return similarity_graph


if __name__ == "__main__":
    graph = generate_graph()
    
    print("\n" + "="*80)
    print("SIMILARITY GRAPH (threshold > 0.8)")
    print("="*80)
    print(json.dumps(graph, indent=2))