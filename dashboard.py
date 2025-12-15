"""
Unified Dashboard for ESN Analysis
Combines three visualization tools:
- ESN Graph Visualization (analysis/esn-vis.py)
- Entity Collapse Visualization (rag/rag-vis.py)
- Graph Image Search Demo (rag/demo.py)
"""

import streamlit as st
import sys
from pathlib import Path
import ast
import re

import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import networkx as nx
import base64
import json
from io import BytesIO
import os
import math
from rag.diffg import graphdiff, jaccard
from PIL import Image
from analysis.jaccard_matrix import jaccard as jcd
from analysis.jaccard_matrix import loadgraph as loadgraphset
from rag.graph import loadgraph, entity_collapse, query
from rag.diffg import matrixdiff
from openai import OpenAI


# Add subdirectories to path for imports
sys.path.insert(0, str(Path(__file__).parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent / "rag"))

# Page configuration
st.set_page_config(
    page_title="ESN Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ESN Analysis Dashboard")
st.markdown("Unified visualization and analysis platform for Entity Scene Networks")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 ESN Graph Visualization",
    "🔍 Entity Collapse Visualization", 
    "🔎 Graph Image Search",
    "LMM + RAG Chat",
    "📊 Stats View"
])

# Tab 1: ESN Graph Visualization
with tab1:    
    st.header("🔗 ESN Graph Visualization")
    st.markdown("Interactive visualization of image relationships as a network graph")
    
    @st.cache_data
    def load_tsv_data_esn(tsv_path):
        """Load the TSV data and create a graph."""
        try:
            df = pd.read_csv(tsv_path, sep='\t', header=None, 
                             names=['node1', 'node2', 'connected', 'edge_label'])
            return df
        except FileNotFoundError:
            st.error(f"File not found: {tsv_path}")
            return None
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def create_network_graph_esn(df):
        """Create a NetworkX graph from the dataframe."""
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['node1'], row['node2'], label=row['edge_label'])
        return G
    
    def get_image_crop_base64_esn(image_path, crop_size=80):
        """Load an image, crop the center, and return as base64."""
        try:
            img = Image.open(image_path)
            width, height = img.size
            left = (width - min(width, height)) // 2
            top = (height - min(width, height)) // 2
            right = left + min(width, height)
            bottom = top + min(width, height)
            
            img_cropped = img.crop((left, top, right, bottom))
            img_cropped = img_cropped.resize((crop_size, crop_size), Image.LANCZOS)
            
            buffer = BytesIO()
            img_cropped.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            return None
    
    def create_plotly_graph_esn(G, image_dir="workdata"):
        """Create an interactive Plotly visualization of the graph with images as nodes."""
        image_dir = Path(image_dir)
        components = list(nx.connected_components(G))
        
        # Layout with proper spacing for multiple components
        if len(components) > 1:
            pos = {}
            components = sorted(components, key=len, reverse=True)
            n_components = len(components)
            grid_cols = math.ceil(math.sqrt(n_components))
            
            component_scales = []
            for component in components:
                n_nodes = len(component)
                scale = max(3.0, math.sqrt(n_nodes) * 2.0)
                component_scales.append(scale)
            
            max_scale = max(component_scales)
            spacing = max_scale * 1.8
            
            for idx, component in enumerate(components):
                subgraph = G.subgraph(component)
                n_nodes = len(component)
                k_value = max(5.0, math.sqrt(n_nodes) * 1.5)
                sub_pos = nx.spring_layout(
                    subgraph, k=k_value, iterations=100, seed=42,
                    scale=component_scales[idx]
                )
                
                row = idx // grid_cols
                col = idx % grid_cols
                offset_x = col * spacing
                offset_y = row * spacing
                
                for node, (x, y) in sub_pos.items():
                    pos[node] = (x + offset_x, y + offset_y)
        else:
            n_nodes = G.number_of_nodes()
            k_value = max(5.0, math.sqrt(n_nodes) * 1.5)
            scale = max(3.0, math.sqrt(n_nodes) * 2.0)
            pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42, scale=scale)
        
        # Create edge traces with colors
        edge_traces = []
        edge_labels_x = []
        edge_labels_y = []
        edge_labels_text = []
        
        unique_labels = list(set(nx.get_edge_attributes(G, 'label').values()))
        color_palette = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
            '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
        ]
        label_colors = {label: color_palette[i % len(color_palette)] 
                        for i, label in enumerate(unique_labels)}
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            label = edge[2].get('label', '')
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color=label_colors.get(label, '#888')),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
            
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            edge_labels_x.append(mid_x)
            edge_labels_y.append(mid_y)
            edge_labels_text.append(label)
        
        # Create invisible node trace for hover
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(size=40, color='rgba(0,0,0,0)', line=dict(width=0)),
            showlegend=False
        )
        
        # Edge label trace
        edge_label_trace = go.Scatter(
            x=edge_labels_x, y=edge_labels_y,
            mode='text',
            text=edge_labels_text,
            textposition='middle center',
            textfont=dict(size=9, color='#555'),
            hoverinfo='text',
            hovertext=edge_labels_text,
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace, edge_label_trace])
        
        # Add images as layout images at node positions
        x_range = max(node_x) - min(node_x) if len(node_x) > 1 else 1
        y_range = max(node_y) - min(node_y) if len(node_y) > 1 else 1
        img_size = min(x_range, y_range) * 0.12
        
        layout_images = []
        for node in G.nodes():
            x, y = pos[node]
            image_path = Path(image_dir) / node
            img_base64 = get_image_crop_base64_esn(image_path)
            if img_base64:
                layout_images.append(dict(
                    source=img_base64,
                    xref="x", yref="y",
                    x=x, y=y,
                    sizex=img_size, sizey=img_size,
                    xanchor="center", yanchor="middle",
                    layer="above"
                ))
        
        fig.update_layout(
            title="Image Relationship Network",
            showlegend=False,
            hovermode='closest',
            dragmode='pan',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                      scaleanchor="y", scaleratio=1),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.5)',
            height=700,
            images=layout_images
        )
        
        return fig, unique_labels, label_colors
    
    # Sidebar settings in columns
    col_settings1, col_settings2 = st.columns([3, 1])
    
    with col_settings1:
        data_dir = Path("data")
        tsv_files = sorted([f for f in data_dir.glob("*.tsv")])
        
        selected_file_esn = st.selectbox(
            "Select TSV file:",
            tsv_files,
            format_func=lambda x: x.name,
            key="esn_file_selector"
        )
    
    with col_settings2:
        image_dir_esn = st.text_input("Image Directory", value="workdata", key="esn_image_dir")
    
    if selected_file_esn:
        df_esn = load_tsv_data_esn(str(selected_file_esn))
        
        if df_esn is not None:
            G_esn = create_network_graph_esn(df_esn)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Nodes", G_esn.number_of_nodes())
            col2.metric("Total Edges", G_esn.number_of_edges())
            edge_labels_esn = set(df_esn['edge_label'].unique())
            col3.metric("Unique Edge Labels", len(edge_labels_esn))
            col4.metric("Connected Components", nx.number_connected_components(G_esn))
            
            # Filter by edge label
            label_options_esn = ["All"] + sorted(edge_labels_esn)
            selected_label_esn = st.selectbox("Filter by Edge Label:", label_options_esn, key="esn_label_filter")
            
            if selected_label_esn != "All":
                filtered_df_esn = df_esn[df_esn['edge_label'] == selected_label_esn]
                G_filtered_esn = create_network_graph_esn(filtered_df_esn)
            else:
                G_filtered_esn = G_esn
                filtered_df_esn = df_esn
            
            # Create tabs
            subtab1, subtab2, subtab3 = st.tabs(["📊 Graph Visualization", "📋 Edge Details", "📈 Statistics"])
            
            with subtab1:
                st.subheader("Interactive Network Graph")
                st.markdown("*Hover over nodes to see connections. Drag to pan, scroll to zoom.*")
                
                if G_filtered_esn.number_of_nodes() > 0:
                    fig_esn, unique_labels_esn, label_colors_esn = create_plotly_graph_esn(
                        G_filtered_esn, image_dir_esn
                    )
                    st.plotly_chart(fig_esn, use_container_width=True, config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'resetScale2d']
                    })
                    
                    # Legend
                    st.subheader("Edge Label Legend")
                    cols = st.columns(min(5, len(unique_labels_esn)))
                    for i, label in enumerate(sorted(unique_labels_esn)):
                        with cols[i % len(cols)]:
                            color = label_colors_esn[label]
                            st.markdown(f"<span style='color:{color}'>■</span> {label}", 
                                       unsafe_allow_html=True)
                else:
                    st.warning("No nodes to display with current filter.")
            
            with subtab2:
                st.subheader("Edge Details")
                st.write(f"Showing {len(filtered_df_esn)} edges")
                
                display_df_esn = filtered_df_esn.copy()
                display_df_esn.columns = ['Node 1', 'Node 2', 'Connected', 'Edge Label']
                st.dataframe(display_df_esn, use_container_width=True)
                
                st.subheader("Edges by Label")
                for label in sorted(edge_labels_esn):
                    label_edges = df_esn[df_esn['edge_label'] == label]
                    with st.expander(f"{label} ({len(label_edges)} edges)"):
                        for _, row in label_edges.iterrows():
                            st.write(f"• {row['node1']} ↔ {row['node2']}")
            
            with subtab3:
                st.subheader("Graph Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Node Degree Distribution**")
                    degree_data = dict(G_esn.degree())
                    degree_df = pd.DataFrame({
                        'Node': list(degree_data.keys()),
                        'Connections': list(degree_data.values())
                    }).sort_values('Connections', ascending=False)
                    st.dataframe(degree_df, use_container_width=True)
                
                with col2:
                    st.write("**Edge Label Distribution**")
                    label_counts = df_esn['edge_label'].value_counts()
                    label_df = pd.DataFrame({
                        'Label': label_counts.index,
                        'Count': label_counts.values
                    })
                    st.dataframe(label_df, use_container_width=True)
                
                st.subheader("Graph Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Graph Density", f"{nx.density(G_esn):.4f}")
                    if not nx.is_connected(G_esn):
                        st.caption("Graph is not fully connected")
                
                with col2:
                    avg_degree = sum(dict(G_esn.degree()).values()) / G_esn.number_of_nodes()
                    st.metric("Average Degree", f"{avg_degree:.2f}")
                
                with col3:
                    components = list(nx.connected_components(G_esn))
                    st.metric("Connected Components", len(components))
        else:
            st.warning("Please provide a valid TSV file.")
            st.info("""
            Expected TSV format (tab-separated, no header):
            ```
            Node1    Node2    Connected    EdgeLabel
            ```
            """)

# Tab 2: Entity Collapse Visualization
with tab2:
    st.header("Entity Collapse Visualization")
    st.markdown("View clustered entity groups from graph data")
    
    data_dir = Path("data")
    tsv_files2 = sorted([f for f in data_dir.glob("*.tsv")])
    
    selected_file2 = st.selectbox(
        "Select a TSV file:",
        tsv_files2,
        format_func=lambda x: x.name,
        key="entity_file_selector"
    )
    
    if selected_file2:
        with st.spinner("Loading and processing graph..."):
            graph = loadgraph(str(selected_file2))
            entities = entity_collapse(graph, clustering_tr=0.5)
        
        st.success(f"Loaded {len(graph)} edges, collapsed into {len(entities)} entities")
        
        st.header("Entity Groups")
        
        for entity_label, image_list in entities:
            st.subheader(entity_label)
            
            cols = st.columns(min(len(image_list), 6))
            
            for idx, img_name in enumerate(image_list):
                col_idx = idx % 6
                img_path = Path("workdata") / img_name
                
                with cols[col_idx]:
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((200, 200))
                            st.image(img, caption=img_name, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading {img_name}: {e}")
                    else:
                        st.warning(f"{img_name} not found")
            
            st.markdown("---")

# Tab 3: Graph Image Search
with tab3:
    
    st.header("Graph Image Search Demo")
    st.markdown("Search for images across multiple graph datasets")
    
    term = st.text_input("Search Term", value="Downtown", key="search_term")
    
    with st.spinner("Loading graphs..."):
        graphs = [
            entity_collapse(loadgraph("data/F1.tsv")), 
            entity_collapse(loadgraph("data/F2.tsv")), 
            entity_collapse(loadgraph("data/F3.tsv")), 
            entity_collapse(loadgraph("data/U1.tsv")),
            entity_collapse(loadgraph("data/U2.tsv")),
            entity_collapse(loadgraph("data/U3.tsv"))
        ]
    
    if st.button("Search", key="search_button"):
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        grid_cols = [col1, col2, col3, col4, col5, col6]
        
        results = [
            query(graphs[0], term),
            query(graphs[1], term),
            query(graphs[2], term),
            query(graphs[3], term),
            query(graphs[4], term),
            query(graphs[5], term)
        ]
        
        dataset_labels = ["F1.tsv", "F2.tsv", "F3.tsv", "U1.tsv", "U2.tsv", "U3.tsv"]
        
        for i, (col, result, label) in enumerate(zip(grid_cols, results, dataset_labels)):
            with col:
                st.subheader(label)
                if result:
                    edge_label, images = result
                    st.write(f"**Match:** {edge_label}")
                    st.write(f"**{len(images)} images found**")
                    
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

# Tab 4: LMM + RAG Chat
with tab4:
    client = OpenAI(
        base_url="http://10.42.0.11:18181/v1",
        api_key="not-needed"
    )
    MODEL_NAME = "NexaAI/Qwen3-VL-4B-Instruct-GGUF"
    
    def encode_image_base64(image_path):
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    st.header("🤖 LMM + RAG Chat")
    st.markdown("Query graphs with natural language and get LMM responses with image context")
    st.info("🖥️ Using local model: NexaAI/Qwen3-VL-4B-Instruct-GGUF at http://10.42.0.11:18181")
    
    st.subheader("Select Graphs to Query")
    col1, col2 = st.columns(2)
    
    data_dir = Path("data")
    available_datasets = ["F1", "F2", "F3", "F4", "U1", "U2", "U3", "U4"]
    
    with col1:
        graph1_name = st.selectbox("Graph 1", available_datasets, index=0, key="chat_graph1")
    with col2:
        graph2_name = st.selectbox("Graph 2", available_datasets, index=4, key="chat_graph2")
    
    # User prompt input
    st.subheader("Enter Your Query")
    user_prompt = st.text_area(
        "Prompt",
        placeholder="e.g., 'What buildings or locations do you see?'",
        height=100,
        key="chat_prompt"
    )
    
    if st.button("Send", type="primary", disabled=not user_prompt):
        if not user_prompt:
            st.error("Please enter a prompt")
        else:
            try:
                
                # Load and collapse graphs
                with st.spinner("Loading graphs..."):
                    graph1_path = f"data/{graph1_name}.tsv"
                    graph2_path = f"data/{graph2_name}.tsv"
                    
                    graph1 = entity_collapse(loadgraph(graph1_path))
                    graph2 = entity_collapse(loadgraph(graph2_path))
                
                col1, col2, col3 = st.columns(3)
                
                # Query Graph 1
                with col1:
                    st.subheader(f"📊 {graph1_name} (with RAG)")
                    with st.spinner(f"Querying {graph1_name}..."):
                        result1 = query(graph1, user_prompt)
                        
                        if result1:
                            _, image_files1 = result1
                            
                            if image_files1:
                                rag_prompt = f"You are shown images related to '{user_prompt}'. Answer the following using the provided images: {user_prompt}"
                                
                                # Create message with base64-encoded images
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": rag_prompt}]
                                    }
                                ]
                                
                                for img_file in image_files1:
                                    img_path = Path("workdata") / img_file
                                    if img_path.exists():
                                        try:
                                            img_base64 = encode_image_base64(str(img_path))
                                            messages[0]["content"].append({
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                                            })
                                        except Exception:
                                            pass
                                
                                response1 = client.chat.completions.create(
                                    model=MODEL_NAME,
                                    messages=messages,
                                    max_tokens=300,
                                    temperature=0.7
                                )
                                
                                st.markdown("**Response:**")
                                st.write(response1.choices[0].message.content)
                                
                                st.markdown(f"**Retrieved Images ({len(image_files1)}):**")
                                img_cols = st.columns(min(len(image_files1), 4))
                                for idx, img_file in enumerate(image_files1[:4]):
                                    img_path = Path("workdata") / img_file
                                    if img_path.exists():
                                        with img_cols[idx % 4]:
                                            st.image(str(img_path), use_container_width=True)
                                if len(image_files1) > 4:
                                    st.caption(f"...and {len(image_files1) - 4} more images")
                            else:
                                st.warning("No valid images found for this query")
                        else:
                            st.warning("No matches found in graph")
                
                # Query Graph 2
                with col2:
                    st.subheader(f"📊 {graph2_name} (with RAG)")
                    with st.spinner(f"Querying {graph2_name}..."):
                        result2 = query(graph2, user_prompt)
                        
                        if result2:
                            _, image_files2 = result2
                            
                            if image_files2:
                                rag_prompt = f"You are shown images related to '{user_prompt}'. Answer the following using the provided images: {user_prompt}"
                                
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": rag_prompt}]
                                    }
                                ]
                                
                                for img_file in image_files2:
                                    img_path = Path("workdata") / img_file
                                    if img_path.exists():
                                        try:
                                            img_base64 = encode_image_base64(str(img_path))
                                            messages[0]["content"].append({
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                                            })
                                        except Exception:
                                            pass
                                
                                response2 = client.chat.completions.create(
                                    model=MODEL_NAME,
                                    messages=messages,
                                    max_tokens=300,
                                    temperature=0.7
                                )
                                
                                st.markdown("**Response:**")
                                st.write(response2.choices[0].message.content)
                                
                                st.markdown(f"**Retrieved Images ({len(image_files2)}):**")
                                img_cols = st.columns(min(len(image_files2), 4))
                                for idx, img_file in enumerate(image_files2[:4]):
                                    img_path = Path("workdata") / img_file
                                    if img_path.exists():
                                        with img_cols[idx % 4]:
                                            st.image(str(img_path), use_container_width=True)
                                if len(image_files2) > 4:
                                    st.caption(f"...and {len(image_files2) - 4} more images")
                            else:
                                st.warning("No valid images found for this query")
                        else:
                            st.warning("No matches found in graph")
                
                # Baseline (no RAG)
                with col3:
                    st.subheader("🔮 Baseline (no RAG)")
                    with st.spinner("Generating baseline response..."):
                        baseline_response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": user_prompt}],
                            max_tokens=300,
                            temperature=0.7
                        )
                        
                        st.markdown("**Response:**")
                        st.write(baseline_response.choices[0].message.content)
                        
                        st.caption("No images provided - pure LLM response")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the local model server at http://10.42.0.11:18181 is running")
    
    # Information section
    with st.expander("How it works"):
        st.markdown("""
        **LMM + RAG Chat** combines Large Multimodal Models with Retrieval-Augmented Generation:
        
        1. **Enter your prompt** - Ask a question or describe what you're looking for
        2. **Select graphs** - Choose two different graph datasets to query
        3. **RAG retrieval** - Your prompt is used to query each graph's entity-collapsed representation
        4. **Image context** - Retrieved images are sent to the LLM along with your prompt
        5. **LMM response** - Gemini generates responses based on the visual context
        6. **Baseline comparison** - See what the LLM says without any image context
        
        **Note:** This is a single-turn chat. Each new message creates a fresh context.
        """)


# Tab 5: Stats View
with tab5:
    
    st.header("📊 Stats View")
    st.markdown("Comparison matrices showing Jaccard distances and graph differences")

    def jaccard_matrix(graph_paths):
        """Generate Jaccard distance matrix for multiple graphs."""
        graphs = [loadgraphset(path) for path in graph_paths]
        n = len(graphs)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            matrix[i][i] = 1.0
        
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = jcd(graphs[i], graphs[j])
                # Mirror to lower triangle
                matrix[j][i] = matrix[i][j]
        
        return matrix
    
    # Dataset labels
    datasets = ["F1", "F2", "F3", "F4", "U1", "U2", "U3", "U4"]
    graph_paths = [f"data/{ds}.tsv" for ds in datasets]
    
    # Add cache control
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Table 1: Jaccard Distance Matrix")
        st.markdown("Based on edge-level Jaccard similarity between graphs")
    with col2:
        if st.button("🔄 Clear Cache", help="Clear cached results and recalculate"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    with st.spinner("Computing Jaccard matrix..."):
        jaccard_mat = jaccard_matrix(graph_paths)
        jaccard_df = pd.DataFrame(jaccard_mat, index=datasets, columns=datasets)
    
    # Display heatmap for Jaccard matrix
    # Custom colorscale: <0.2=red, 0.2-0.3=yellow, >0.3=green (continuous gradient)
    custom_colorscale = [
        [0.0, 'rgb(255, 0,0)'],      # Red at 0
        [0.4, 'rgb(200,255,0)'],      # Green at 0.4
        [1.0, 'rgb(0,200,0)']       # Dark green at 1.0
    ]
    
    fig1 = go.Figure(data=go.Heatmap(
        z=jaccard_df.values,
        x=datasets,
        y=datasets,
        colorscale=custom_colorscale,
        zmin=0,
        zmax=1,
        text=np.round(jaccard_df.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Jaccard Index")
    ))
    
    fig1.update_layout(
        title="Jaccard Distance Matrix",
        xaxis_title="Dataset",
        yaxis_title="Dataset",
        height=550,
        width=550
    )
    
    # Calculate statistics for Jaccard matrix
    upper_triangle_indices = np.triu_indices_from(jaccard_df.values, k=1)
    jaccard_values = jaccard_df.values[upper_triangle_indices]
    
    # Create pairwise comparison list
    jaccard_pairs = []
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            jaccard_pairs.append({
                'Dataset 1': datasets[i],
                'Dataset 2': datasets[j],
                'Jaccard Score': jaccard_df.values[i, j]
            })
    
    jaccard_pairs_df = pd.DataFrame(jaccard_pairs).sort_values('Jaccard Score', ascending=False)
    
    # Add group classification and highlighting
    def classify_pair(row):
        d1, d2 = row['Dataset 1'], row['Dataset 2']
        both_f = d1.startswith('F') and d2.startswith('F')
        both_u = d1.startswith('U') and d2.startswith('U')
        if both_f or both_u:
            return 'Within Group'
        return 'Between Groups'
    
    jaccard_pairs_df['Group Type'] = jaccard_pairs_df.apply(classify_pair, axis=1)
    jaccard_pairs_df = jaccard_pairs_df[['Dataset 1', 'Dataset 2', 'Group Type', 'Jaccard Score']]
    
    def highlight_groups(row):
        if row['Group Type'] == 'Within Group':
            return ['background-color: lightgreen'] * len(row)
        else:
            return ['background-color: lightcoral'] * len(row)
    
    # Calculate ANOSIM R for Jaccard matrix
    def calculate_anosim_r(similarity_matrix, datasets):
        """Calculate ANOSIM R: (mean_rank_between - mean_rank_within) / (N/2)"""
        pairs = []
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                d1, d2 = datasets[i], datasets[j]
                both_f = d1.startswith('F') and d2.startswith('F')
                both_u = d1.startswith('U') and d2.startswith('U')
                within_group = both_f or both_u
                
                # Convert similarity to dissimilarity/distance
                distance = 1 - similarity_matrix[i, j]
                pairs.append({
                    'distance': distance,
                    'within_group': within_group
                })
        
        # Sort by distance and assign ranks (1 = smallest distance)
        pairs_sorted = sorted(pairs, key=lambda x: x['distance'])
        for rank, pair in enumerate(pairs_sorted, start=1):
            pair['rank'] = rank
        
        # Calculate mean ranks
        within_ranks = [p['rank'] for p in pairs if p['within_group']]
        between_ranks = [p['rank'] for p in pairs if not p['within_group']]
        
        if not within_ranks or not between_ranks:
            return 0.0
        
        r_W = np.mean(within_ranks)
        r_B = np.mean(between_ranks)
        N = len(pairs)
        
        # ANOSIM R = (r_B - r_W) / (N/2)
        R = (r_B - r_W) / (N / 2)
        
        return R
    
    anosim_r_jaccard = calculate_anosim_r(jaccard_df.values, datasets)
    
    # Display in two columns: heatmap on left, table on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig1, use_container_width=False)
    
    with col2:
        # Statistics
        st.metric("ANOSIM R", f"{anosim_r_jaccard:.3f}")
        if anosim_r_jaccard > 0.75:
            st.caption("🟢 Strong separation between groups")
        elif anosim_r_jaccard > 0.5:
            st.caption("🟡 Moderate separation between groups")
        elif anosim_r_jaccard > 0.25:
            st.caption("🟠 Weak separation between groups")
        else:
            st.caption("🔴 Little to no separation between groups")
        
        st.markdown("**Statistics:**")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.write(f"Mean: {jaccard_values.mean():.3f}")
            st.write(f"Median: {np.median(jaccard_values):.3f}")
        with metric_col2:
            st.write(f"Std Dev: {jaccard_values.std():.3f}")
        
        st.markdown("**Pairwise Rankings:**")
        styled_jaccard = jaccard_pairs_df.style.apply(highlight_groups, axis=1)
        st.dataframe(styled_jaccard, use_container_width=True, hide_index=True, height=400)
    
    st.markdown("---")
    
    # Table 2: Graph Diff Matrix
    st.subheader("Table 2: Graph Difference Matrix")
    st.markdown("Based on semantic edge label similarity using RAG query comparison")
    
    @st.cache_data
    def getMat(graph_paths):
        # cache
        return matrixdiff(graph_paths)


    with st.spinner("Computing graph difference matrix (this may take a while)..."):
        detail, diff_mat = getMat(graph_paths)
        diff_df = pd.DataFrame(diff_mat, index=datasets, columns=datasets)
        
    # Display heatmap for Diff matrix
    fig2 = go.Figure(data=go.Heatmap(
        z=diff_df.values,
        x=datasets,
        y=datasets,
        colorscale=custom_colorscale,
        zmin=0,
        zmax=1,
        text=np.round(diff_df.values, 3),
        hoverinfo='text',
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Similarity Score")
    ))
    
    fig2.update_layout(
        title="Graph Difference Matrix",
        xaxis_title="Dataset",
        yaxis_title="Dataset",
        height=550,
        width=550
    )
    
    # Calculate statistics for Diff matrix
    diff_upper_triangle = np.triu_indices_from(diff_df.values, k=1)
    diff_values = diff_df.values[diff_upper_triangle]
    
    # Create pairwise comparison list
    diff_pairs = []
    for i in range(len(datasets)):
        for j in range(i+1, len(datasets)):
            diff_pairs.append({
                'Dataset 1': datasets[i],
                'Dataset 2': datasets[j],
                'Similarity Score': diff_df.values[i, j]
            })
    
    diff_pairs_df = pd.DataFrame(diff_pairs).sort_values('Similarity Score', ascending=False)
    diff_pairs_df['Group Type'] = diff_pairs_df.apply(classify_pair, axis=1)
    diff_pairs_df = diff_pairs_df[['Dataset 1', 'Dataset 2', 'Group Type', 'Similarity Score']]
    
    # Calculate ANOSIM R for Diff matrix
    anosim_r_diff = calculate_anosim_r(diff_df.values, datasets)
    
    # Display in two columns: heatmap on left, table on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig2, use_container_width=False)
    
    with col2:
        # Statistics
        st.metric("ANOSIM R", f"{anosim_r_diff:.3f}")
        if anosim_r_diff > 0.75:
            st.caption("🟢 Strong separation between groups")
        elif anosim_r_diff > 0.5:
            st.caption("🟡 Moderate separation between groups")
        elif anosim_r_diff > 0.25:
            st.caption("🟠 Weak separation between groups")
        else:
            st.caption("🔴 Little to no separation between groups")
        
        st.markdown("**Statistics:**")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.write(f"Mean: {diff_values.mean():.3f}")
            st.write(f"Median: {np.median(diff_values):.3f}")
        with metric_col2:
            st.write(f"Std Dev: {diff_values.std():.3f}")
        
        st.markdown("**Pairwise Rankings:**")
        styled_diff = diff_pairs_df.style.apply(highlight_groups, axis=1)
        st.dataframe(styled_diff, use_container_width=True, hide_index=True, height=400)
    
    st.markdown("---")
    
    # Table 3: Gemini Response Reranking Similarity Matrix
    st.subheader("Table 3: LMM Response Semantic Similarity Matrix")
    st.markdown("Based on MiniLM-L6-v2 text embeddings of raw outputs with cosine similarity")
    
    # Dynamic edge label selection with checkboxes
    st.markdown("**Select Edge Labels to Include:**")
    
    # Available edge labels (from gemini-run.py)
    available_edge_labels = ["bridge", "trail", "city", "library", "Iowa Memorial Union (IMU)", "Seamans engineering"]
    
    # Create checkboxes in columns
    checkbox_cols = st.columns(3)
    selected_edge_labels = []
    
    for idx, edge_label in enumerate(available_edge_labels):
        col_idx = idx % 3
        with checkbox_cols[col_idx]:
            if st.checkbox(edge_label, value=(edge_label in ["library", "Iowa Memorial Union (IMU)"]), key=f"edge_checkbox_{edge_label}"):
                selected_edge_labels.append(edge_label)
    
    if not selected_edge_labels:
        st.warning("Please select at least one edge label to generate the similarity matrix.")
        st.stop()
    
    def gemini_rerank_matrix_multi(datasets, edge_labels):
        """Generate similarity matrix for Gemini outputs using MiniLM embeddings and cosine similarity.
        
        Creates embeddings for the raw text output and computes cosine similarity.
        """
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Track which datasets have all required outputs
        valid_datasets = []
        all_raw_texts_by_label = {label: [] for label in edge_labels}  # Store raw text content
        
        for dataset in datasets:
            has_all_labels = True
            dataset_raw_texts = {}
            
            for edge_label in edge_labels:
                output_path = Path(f"data/gemini/{dataset}/{edge_label}.out")
                if output_path.exists():
                    with open(output_path, 'r') as f:
                        text = f.read().strip()
                        if text:
                            dataset_raw_texts[edge_label] = text
                        else:
                            has_all_labels = False
                            st.warning(f"Empty output file for {dataset}/{edge_label}.out")
                            break
                else:
                    has_all_labels = False
                    st.warning(f"Missing output file: {output_path}")
                    break
            
            if has_all_labels:
                valid_datasets.append(dataset)
                for label in edge_labels:
                    all_raw_texts_by_label[label].append(dataset_raw_texts[label])
        
        if len(valid_datasets) < 2:
            st.error("Not enough valid Gemini outputs found. Run gemini-run.py first for the selected edge labels.")
            return None, None, None, None
        
        n = len(valid_datasets)
        
        # Compute similarity matrices for each edge label
        similarity_matrices = []
        
        for edge_label in edge_labels:
            texts = all_raw_texts_by_label[edge_label]
            
            # Generate embeddings for raw texts
            embeddings = model.encode(texts, convert_to_tensor=False)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            similarity_matrices.append(similarity_matrix)
        
        # Average across all edge labels
        avg_similarity_matrix = np.mean(similarity_matrices, axis=0)
        for i in range(len(avg_similarity_matrix)):
            for j in range(len(avg_similarity_matrix[i])):
                avg_similarity_matrix[i][j] = float("{:.2f}".format(avg_similarity_matrix[i][j]))
        
        print(avg_similarity_matrix)

        # Return all data needed for detailed view
        return avg_similarity_matrix, valid_datasets, all_raw_texts_by_label, similarity_matrices
    
    with st.spinner(f"Computing Gemini response similarity matrix across {', '.join(selected_edge_labels)}..."):
        result = gemini_rerank_matrix_multi(datasets, edge_labels=selected_edge_labels)
    
    if result[0] is not None:
        gemini_mat, valid_datasets, all_raw_texts_by_label, similarity_matrices = result
        print(gemini_mat)

        # Convert to generic 2D array with rounded values
        simMat = [[round(gemini_mat[i][j].item(), 2) for j in range(len(gemini_mat[i]))] for i in range(len(gemini_mat))]

        print(simMat)

        gemini_df = pd.DataFrame(simMat, index=valid_datasets, columns=valid_datasets)
        edge_labels = selected_edge_labels
        
        # Create simplified hover text
        hover_text = []
        for i in range(len(valid_datasets)):
            row_hover = []
            for j in range(len(valid_datasets)):
                if i == j:
                    hover_text_cell = f"<b>{valid_datasets[i]}</b><br>Avg Similarity: 1.00<br>(Same dataset)<br><br>Click for details"
                else:
                    hover_text_cell = (
                        f"<b>{valid_datasets[i]} vs {valid_datasets[j]}</b><br>"
                        f"Avg Similarity: {gemini_df.values[i][j]:.2f}<br><br>"
                        f"Click to see detailed comparison"
                    )
                row_hover.append(hover_text_cell)
            hover_text.append(row_hover)
        
        # Create heatmap with custom colorscale for Table 3
        gemini_colorscale = [
            [0.0, 'rgb(255, 0, 0)'],      
            [0.5, 'rgb(255, 255, 0)'],    
            [1.0, 'rgb(0, 200, 0)']     
        ]
        
        fig3 = go.Figure(data=go.Heatmap(
            z=gemini_df.values,
            x=valid_datasets,
            y=valid_datasets,
            colorscale=gemini_colorscale,
            zmin=0,
            zmax=1,
            text=gemini_df.values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertext=hover_text,
            hoverlabel=dict(bgcolor="black"),
            hoverinfo='text',
            colorbar=dict(title="Cosine Similarity Score")
        ))
        
        # Create dynamic title based on selected edge labels
        edge_labels_str = " + ".join(edge_labels)
        fig3.update_layout(
            title=f"Gemini Response Similarity Matrix ({edge_labels_str} avg)",
            xaxis_title="Dataset",
            yaxis_title="Dataset",
            height=550,
            width=550,
            hoverlabel=dict(
                bgcolor="white",
                font_size=10,
                font_family="monospace",
                align="left",
                namelength=-1
            )

        )
        
        # Calculate statistics
        upper_triangle_indices = np.triu_indices_from(gemini_df.values, k=1)
        gemini_values = gemini_df.values[upper_triangle_indices]
        
        # Create pairwise comparison list
        gemini_pairs = []
        for i in range(len(valid_datasets)):
            for j in range(i + 1, len(valid_datasets)):
                ds1 = valid_datasets[i]
                ds2 = valid_datasets[j]
                score = gemini_df.values[i][j]
                
                if ds1[0] == ds2[0]:
                    group_type = "Within-group"
                else:
                    group_type = "Between-group"
                
                gemini_pairs.append({
                    'Dataset 1': ds1,
                    'Dataset 2': ds2,
                    'Group Type': group_type,
                    'Similarity Score': score
                })
        
        gemini_pairs_df = pd.DataFrame(gemini_pairs)
        gemini_pairs_df = gemini_pairs_df.sort_values('Similarity Score', ascending=False)
        gemini_pairs_df = gemini_pairs_df[['Dataset 1', 'Dataset 2', 'Group Type', 'Similarity Score']]
        
        # Calculate ANOSIM R
        anosim_r_gemini = calculate_anosim_r(gemini_df.values, valid_datasets)
        
        # Display in two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Use plotly_events to capture clicks
            st.plotly_chart(fig3, use_container_width=False, key="gemini_heatmap")
        
        with col2:
            # Statistics
            st.metric("ANOSIM R", f"{anosim_r_gemini:.3f}")
            if anosim_r_gemini > 0.75:
                st.caption("🟢 Strong separation between groups")
            elif anosim_r_gemini > 0.5:
                st.caption("🟡 Moderate separation between groups")
            elif anosim_r_gemini > 0.25:
                st.caption("🟠 Weak separation between groups")
            else:
                st.caption("🔴 Little to no separation between groups")
            
            st.markdown("**Statistics:**")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.write(f"Mean: {gemini_values.mean():.2f}")
                st.write(f"Median: {np.median(gemini_values):.2f}")
            with metric_col2:
                st.write(f"Std Dev: {gemini_values.std():.2f}")
            
            st.markdown("**Pairwise Rankings:**")
            # Format similarity scores to 2 decimal places for display
            gemini_pairs_df['Similarity Score'] = gemini_pairs_df['Similarity Score'].apply(lambda x: f"{x:.2f}")
            styled_gemini = gemini_pairs_df.style.apply(highlight_groups, axis=1)
            st.dataframe(styled_gemini, use_container_width=True, hide_index=True, height=400)
        
        # Add selection interface
        st.markdown("---")
        st.subheader("Detailed Comparison")
        st.markdown(f"Select two datasets to compare their responses across {', '.join(edge_labels)} queries")
        
        col1, col2 = st.columns(2)
        with col1:
            dataset1_select = st.selectbox("Dataset 1", valid_datasets, key="gemini_dataset1")
        with col2:
            dataset2_select = st.selectbox("Dataset 2", valid_datasets, index=min(1, len(valid_datasets)-1), key="gemini_dataset2")
        
        if dataset1_select and dataset2_select:
            idx1 = valid_datasets.index(dataset1_select)
            idx2 = valid_datasets.index(dataset2_select)
            
            st.markdown(f"### {dataset1_select} vs {dataset2_select}")
            
            # Show average score
            avg_score = gemini_df.values[idx1][idx2]
            st.metric("Average Similarity Score", f"{avg_score:.3f}")
            
            # Create comparison table for each edge label
            for label_idx, edge_label in enumerate(edge_labels):
                st.markdown(f"#### {edge_label.upper()}")
                
                # Get individual score
                individual_score = similarity_matrices[label_idx][idx1][idx2]
                st.write(f"**Similarity Score:** {individual_score:.3f}")
                
                # Display both responses side by side
                resp_col1, resp_col2 = st.columns(2)
                
                with resp_col1:
                    st.markdown(f"**{dataset1_select} Response:**")
                    raw_text = all_raw_texts_by_label[edge_label][idx1]
                    
                    st.text_area(
                        f"{dataset1_select}_{edge_label}_full",
                        raw_text,
                        height=300,
                        key=f"resp_{dataset1_select}_{edge_label}_full",
                        label_visibility="collapsed"
                    )
                
                with resp_col2:
                    st.markdown(f"**{dataset2_select} Response:**")
                    raw_text = all_raw_texts_by_label[edge_label][idx2]
                    
                    st.text_area(
                        f"{dataset2_select}_{edge_label}_full",
                        raw_text,
                        height=300,
                        key=f"resp_{dataset2_select}_{edge_label}_full",
                        label_visibility="collapsed"
                    )
                
                st.markdown("---")
    
    # ANOSIM Interpretation
    st.markdown("---")
    st.write("""
    **ANOSIM R Interpretation:**
    - R close to 1: Groups are well separated (dissimilar)
    - R close to 0: No difference between groups
    - R < 0: Within-group dissimilarity exceeds between-group dissimilarity
    """)