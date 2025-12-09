## This is a rather expensive full test suite
## Query each graph with predefined edge labels and send all images to Gemini

import os
from google import genai
from rag.graph import loadgraph, entity_collapse, query
from PIL import Image
from pathlib import Path

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GAPI"])

data = ["U1.tsv", "U2.tsv", "U3.tsv", "U4.tsv", 
        "F1.tsv", "F2.tsv", "F3.tsv", "F4.tsv"]

# Load all graphs
graphs = []
for filename in data:
    path = Path("data") / filename
    graphs.append(loadgraph(str(path)))

collapsed_graphs = []
for i, graph in enumerate(graphs):
    print(f"Collapsing graph {i+1}/8...")
    collapsed_graphs.append(entity_collapse(graph, clustering_tr=0.5))

edgelist = ["very dark"]

# Create output directories
Path("data/gemini").mkdir(exist_ok=True)
for dataset_name in data:
    dataset_prefix = dataset_name.replace(".tsv", "")
    Path(f"data/gemini/{dataset_prefix}").mkdir(exist_ok=True)

for edge in edgelist:
    print(f"\nProcessing edge: {edge}")
    
    for idx, collapsed_graph in enumerate(collapsed_graphs):
        dataset_name = data[idx].replace(".tsv", "")
        print(f"  Querying {dataset_name}...")
        
        result = query(collapsed_graph, edge)
        
        if not result:
            print(f"    No matches found for {edge} in {dataset_name}")
            continue
        
        _, image_files = result
        
        # Load all images
        images = []
        for img_file in image_files:
            img_path = Path("workdata") / img_file
            if img_path.exists():
                try:
                    images.append(Image.open(img_path))
                except Exception as e:
                    print(f"    Warning: Could not load {img_file}: {e}")
        
        if not images:
            print(f"    No valid images found for {edge} in {dataset_name}")
            continue
        
        # Create prompt and send to Gemini
        prompt = f"You are shown images that you associate to '{edge}'. Answer the following in detail: What do you think is associated with '{edge}'?"
        
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[prompt, *images]
            )
            
            # Save output
            output_path = Path(f"data/gemini/{dataset_name}/{edge}.out")
            with open(output_path, "w") as f:
                f.write(response.text)
            
            print(f"    ✓ Saved to {output_path}")
            
        except Exception as e:
            print(f"    Error calling Gemini: {e}")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Results saved to data/gemini/[U1-U4,F1-F4]/[edge].out")
        