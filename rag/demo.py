import streamlit as st
import os
import sys

from graph import loadgraph, entity_collapse, query
st.set_page_config(layout="wide", page_title="Graph Image Search")
st.title("Graph Image Search Demo")

# Input field
term = st.text_input("Term field", value="Downtown")

graphs = [
    entity_collapse(loadgraph("data/F1.tsv")), 
    entity_collapse(loadgraph("data/F2.tsv")), 
    entity_collapse(loadgraph("data/F3.tsv")), 
    entity_collapse(loadgraph("data/U1.tsv"))
]

# Button to trigger search
if st.button("Go"):
    # Create a 2x2 grid layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    grid_cols = [col1, col2, col3, col4]

    # Query each graph
    results = [
        query(graphs[0], term),
        query(graphs[1], term),
        query(graphs[2], term),
        query(graphs[3], term)
    ]
    
    # Dataset labels
    dataset_labels = ["F1.tsv", "F2.tsv", "F3.tsv", "U1.tsv", "U2.tsv", "U3.tsv"]
    
    # Display results in each column
    for i, (col, result, label) in enumerate(zip(grid_cols, results, dataset_labels)):
        with col:
            st.subheader(label)
            if result:
                edge_label, images = result
                st.write(f"**Match:** {edge_label}")
                st.write(f"**{len(images)} images found**")
                
                # Create scrollable container with horizontal images
                with st.container(height=300):
                    img_cols = st.columns(len(images))
                    for idx, img_name in enumerate(images):
                        with img_cols[idx]:
                            img_path = os.path.join("workdata", img_name)
                            if os.path.exists(img_path):
                                st.image(img_path, caption=img_name, width=150)
                            else:
                                st.text(f"Not found: {img_name}")
            else:
                st.write("No results found")
    
