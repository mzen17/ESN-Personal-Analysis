import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="ESN Graph Visualization",
    page_icon="🔗",
    layout="wide"
)

# Title and description
st.title("🔗 ESN Graph Visualization")
st.markdown("Interactive visualization of image relationships as a network graph")


@st.cache_data
def load_tsv_data(tsv_path):
    """Load the TSV data and create a graph."""
    try:
        # Read TSV file (tab-separated, no header)
        df = pd.read_csv(tsv_path, sep='\t', header=None, 
                         names=['node1', 'node2', 'connected', 'edge_label'])
        return df
    except FileNotFoundError:
        st.error(f"File not found: {tsv_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def create_network_graph(df):
    """Create a NetworkX graph from the dataframe."""
    G = nx.Graph()
    
    # Add edges with labels
    for _, row in df.iterrows():
        G.add_edge(row['node1'], row['node2'], label=row['edge_label'])
    
    return G


def get_image_crop_base64(image_path, crop_size=80):
    """Load an image, crop the center, and return as base64."""
    try:
        img = Image.open(image_path)
        # Get center crop
        width, height = img.size
        left = (width - min(width, height)) // 2
        top = (height - min(width, height)) // 2
        right = left + min(width, height)
        bottom = top + min(width, height)
        
        # Crop to square from center
        img_cropped = img.crop((left, top, right, bottom))
        # Resize to thumbnail
        img_cropped = img_cropped.resize((crop_size, crop_size), Image.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        img_cropped.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        return None


def create_plotly_graph(G, image_dir="workdata"):
    """Create an interactive Plotly visualization of the graph."""
    
    import math
    
    # Ensure image_dir is a Path object
    image_dir = Path(image_dir)
    
    # Detect connected components (islands)
    components = list(nx.connected_components(G))
    
    # If multiple components, layout each separately and position them in a grid
    if len(components) > 1:
        pos = {}
        
        # Sort components by size (largest first)
        components = sorted(components, key=len, reverse=True)
        
        # Calculate grid layout for components
        n_components = len(components)
        grid_cols = math.ceil(math.sqrt(n_components))
        grid_rows = math.ceil(n_components / grid_cols)
        
        # Calculate scale factor for each component based on its size
        # Larger components get more space
        component_scales = []
        for component in components:
            n_nodes = len(component)
            # Much larger scale for better node separation
            scale = max(3.0, math.sqrt(n_nodes) * 2.0)
            component_scales.append(scale)
        
        # Dynamic spacing based on largest component
        max_scale = max(component_scales)
        spacing = max_scale * 1.8  # Reduced spacing between islands
        
        for idx, component in enumerate(components):
            # Create subgraph for this component
            subgraph = G.subgraph(component)
            n_nodes = len(component)
            
            # Much higher k value = more spacing between nodes
            # k is the optimal distance between nodes
            k_value = max(5.0, math.sqrt(n_nodes) * 1.5)
            sub_pos = nx.spring_layout(
                subgraph, 
                k=k_value, 
                iterations=100, 
                seed=42,
                scale=component_scales[idx]
            )
            
            # Calculate grid position for this component
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Offset to position this component in the grid
            offset_x = col * spacing
            offset_y = row * spacing
            
            # Add positions with offset
            for node, (x, y) in sub_pos.items():
                pos[node] = (x + offset_x, y + offset_y)
    else:
        # Single component - use enhanced spring layout with better spacing
        n_nodes = G.number_of_nodes()
        k_value = max(5.0, math.sqrt(n_nodes) * 1.5)
        scale = max(3.0, math.sqrt(n_nodes) * 2.0)
        pos = nx.spring_layout(
            G, 
            k=k_value, 
            iterations=100, 
            seed=42,
            scale=scale
        )
    
    # Create edge traces
    edge_traces = []
    edge_labels_x = []
    edge_labels_y = []
    edge_labels_text = []
    
    # Color palette for edge labels
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
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=label_colors.get(label, '#888')),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Calculate midpoint for edge label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_labels_x.append(mid_x)
        edge_labels_y.append(mid_y)
        edge_labels_text.append(label)
    
    # Create node trace (invisible markers for hover interaction)
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    # Invisible node trace for hover/click interaction
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=40,
            color='rgba(0,0,0,0)',  # Invisible
            line=dict(width=0)
        ),
        showlegend=False
    )
    
    # Edge label trace
    edge_label_trace = go.Scatter(
        x=edge_labels_x,
        y=edge_labels_y,
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
    # We need to calculate proper sizing based on the coordinate range
    x_range = max(node_x) - min(node_x) if len(node_x) > 1 else 1
    y_range = max(node_y) - min(node_y) if len(node_y) > 1 else 1
    
    # Image size relative to plot (adjust as needed)
    img_size = min(x_range, y_range) * 0.12
    
    layout_images = []
    for node in G.nodes():
        x, y = pos[node]
        image_path = Path(image_dir) / node
        
        img_base64 = get_image_crop_base64(image_path)
        if img_base64:
            layout_images.append(dict(
                source=img_base64,
                xref="x",
                yref="y",
                x=x,
                y=y,
                sizex=img_size,
                sizey=img_size,
                xanchor="center",
                yanchor="middle",
                layer="above"
            ))
    
    fig.update_layout(
        title="Image Relationship Network",
        showlegend=False,
        hovermode='closest',
        dragmode='pan',  # Enable panning by default (middle-click also works)
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False
        ),
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=700,
        images=layout_images
    )
    
    return fig, unique_labels, label_colors


# Sidebar
st.sidebar.header("Settings")

tsv_file = st.sidebar.text_input(
    "TSV File Path",
    value="data/out1.tsv",
    help="Path to the TSV file with graph data"
)

image_dir = st.sidebar.text_input(
    "Image Directory",
    value="workdata",
    help="Path to the directory containing images"
)

# Load data
df = load_tsv_data(tsv_file)

if df is not None:
    # Create graph
    G = create_network_graph(df)
    
    # Display statistics in sidebar
    st.sidebar.subheader("Graph Statistics")
    st.sidebar.metric("Total Nodes", G.number_of_nodes())
    st.sidebar.metric("Total Edges", G.number_of_edges())
    
    # Get unique edge labels
    edge_labels = set(df['edge_label'].unique())
    st.sidebar.metric("Unique Edge Labels", len(edge_labels))
    
    # Filter by edge label
    st.sidebar.subheader("Filter by Edge Label")
    label_options = ["All"] + sorted(edge_labels)
    selected_label = st.sidebar.selectbox("Select Edge Label", label_options)
    
    # Filter graph if needed
    if selected_label != "All":
        filtered_df = df[df['edge_label'] == selected_label]
        G_filtered = create_network_graph(filtered_df)
    else:
        G_filtered = G
        filtered_df = df
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Graph Visualization", "📋 Edge Details", "📈 Statistics"])
    
    with tab1:
        st.subheader("Interactive Network Graph")
        st.markdown("*Hover over nodes to see connections. Node size indicates number of connections.*")
        
        if G_filtered.number_of_nodes() > 0:
            fig, unique_labels, label_colors = create_plotly_graph(G_filtered, image_dir)
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'resetScale2d']
            })
            
            # Legend for edge colors
            st.subheader("Edge Label Legend")
            cols = st.columns(min(5, len(unique_labels)))
            for i, label in enumerate(sorted(unique_labels)):
                with cols[i % len(cols)]:
                    color = label_colors[label]
                    st.markdown(f"<span style='color:{color}'>■</span> {label}", 
                               unsafe_allow_html=True)
        else:
            st.warning("No nodes to display with current filter.")
    
    with tab2:
        st.subheader("Edge Details")
        
        # Show filtered data
        st.write(f"Showing {len(filtered_df)} edges")
        
        # Display as a table
        display_df = filtered_df.copy()
        display_df.columns = ['Node 1', 'Node 2', 'Connected', 'Edge Label']
        st.dataframe(display_df, use_container_width=True)
        
        # Group by edge label
        st.subheader("Edges by Label")
        for label in sorted(edge_labels):
            label_edges = df[df['edge_label'] == label]
            with st.expander(f"{label} ({len(label_edges)} edges)"):
                for _, row in label_edges.iterrows():
                    st.write(f"• {row['node1']} ↔ {row['node2']}")
    
    with tab3:
        st.subheader("Graph Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Node Degree Distribution**")
            degree_data = dict(G.degree())
            degree_df = pd.DataFrame({
                'Node': list(degree_data.keys()),
                'Connections': list(degree_data.values())
            }).sort_values('Connections', ascending=False)
            st.dataframe(degree_df, use_container_width=True)
        
        with col2:
            st.write("**Edge Label Distribution**")
            label_counts = df['edge_label'].value_counts()
            label_df = pd.DataFrame({
                'Label': label_counts.index,
                'Count': label_counts.values
            })
            st.dataframe(label_df, use_container_width=True)
        
        # Graph metrics
        st.subheader("Graph Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if nx.is_connected(G):
                st.metric("Graph Density", f"{nx.density(G):.4f}")
            else:
                st.metric("Graph Density", f"{nx.density(G):.4f}")
                st.caption("Graph is not fully connected")
        
        with col2:
            st.metric("Average Degree", f"{sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        
        with col3:
            components = list(nx.connected_components(G))
            st.metric("Connected Components", len(components))

else:
    st.warning("Please provide a valid TSV file path in the sidebar.")
    st.info("""
    Expected TSV format (tab-separated, no header):
    ```
    Node1    Node2    Connected    EdgeLabel
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built using Streamlit and NetworkX")
