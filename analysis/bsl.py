# baseline script
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from pathlib import Path
import json
import base64
from openai import OpenAI

# Initialize OpenAI client pointing to local server
client = OpenAI(
    base_url="http://10.42.0.11:18181/v1",
    api_key="not-needed"
)

MODEL_NAME = "NexaAI/Qwen3-VL-8B-Instruct-GGUF"

def encode_image_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_relationship_label(img1_path, img2_path):
    """Use AI to generate a label explaining why two images are related."""
    try:
        # Encode both images
        img1_base64 = encode_image_base64(img1_path)
        img2_base64 = encode_image_base64(img2_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are shown two images. Identify what makes these two images similar or related. Describe the common feature, concept, or theme that connects them. Be concise (2-6 words). Output only the relationship label, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img1_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        label = response.choices[0].message.content.strip()
        return label
        
    except Exception as e:
        print(f"Error generating label for {Path(img1_path).name} <-> {Path(img2_path).name}: {e}")
        return "N/A"

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
    
    # Build similarity graph: dict of image -> list of dicts with path and label
    similarity_graph = {}
    
    for i, img1_path in enumerate(valid_paths):
        similar_images = []
        
        for j, img2_path in enumerate(valid_paths):
            if i != j:  # Don't compare image with itself
                # Compute cosine similarity
                similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                
                if similarity > 0.80:
                    # Use AI to generate relationship label
                    print(f"    Generating label for {Path(img1_path).name} <-> {Path(img2_path).name}...")
                    label = get_relationship_label(img1_path, img2_path)
                    similar_images.append({
                        "image": img2_path,
                        "label": label,
                        "similarity": similarity
                    })
        
        similarity_graph[img1_path] = similar_images
        print(f"  {Path(img1_path).name}: {len(similar_images)} similar images")
    
    return similarity_graph


if __name__ == "__main__":
    graph = generate_graph()
    
    print("\n" + "="*80)
    print("SIMILARITY GRAPH (threshold > 0.8)")
    print("="*80)
    print(json.dumps(graph, indent=2))