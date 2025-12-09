import os
from google import genai
from graph import query
from PIL import Image
from pathlib import Path

# Initialize the client with your API key
client = genai.Client(api_key=os.environ["GAPI"])

def ask(graph, term):
    prompt = f"You relate the attached images with {term}. What do you think {term} means?"

    out = query(graph, term)
    if not out:
        return "No matches found."

    _, image_files = out
    images = []
    for img_file in image_files:
        path = Path("workdata") / img_file
        if path.exists():
            try:
                images.append(Image.open(path))
            except Exception:
                pass

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[prompt, *images]
    )
    return response.text
    
