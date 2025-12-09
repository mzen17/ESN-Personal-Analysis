import streamlit as st
from graph import loadgraph, entity_collapse
from PIL import Image
import os
from pathlib import Path

# Set up the Streamlit page
st.set_page_config(page_title="Entity Collapse Visualization", layout="wide")
st.title("Entity Collapse Visualization")

# Get list of TSV files in the data directory
data_dir = Path("data")
tsv_files = sorted([f for f in data_dir.glob("*.tsv") if f.name != "tsv-fix.py"])

# File selector
selected_file = st.selectbox(
    "Select a TSV file:",
    tsv_files,
    format_func=lambda x: x.name
)

if selected_file:
    # Load graph and run entity collapse
    with st.spinner("Loading and processing graph..."):
        graph = loadgraph(str(selected_file))
        entities = entity_collapse(graph, clustering_tr=0.4)
    
    st.success(f"Loaded {len(graph)} edges, collapsed into {len(entities)} entities")
    
    # Display results in a table
    st.header("Entity Groups")
    
    for entity_label, image_list in entities:
        st.subheader(entity_label)
        
        # Create columns for images
        cols = st.columns(min(len(image_list), 6))
        
        for idx, img_name in enumerate(image_list):
            col_idx = idx % 6
            img_path = Path("workdata") / img_name
            
            with cols[col_idx]:
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        # Create thumbnail
                        img.thumbnail((200, 200))
                        st.image(img, caption=img_name, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading {img_name}: {e}")
                else:
                    st.warning(f"{img_name} not found")
        
        st.markdown("---")
