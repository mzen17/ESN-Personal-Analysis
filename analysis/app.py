import streamlit as st
import json
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Image Cluster Visualization",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Image Cluster Visualization")
st.markdown("Interactive visualization of image clusters based on CLIP embeddings")

# Load the JSON data
@st.cache_data
def load_cluster_data(json_path):
    """Load the cluster visualization data from JSON."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {json_path}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON file")
        return None

# Main visualization
def create_scatter_plot(data):
    """Create an interactive scatter plot using Plotly."""
    
    # Prepare data for plotting
    x_coords = []
    y_coords = []
    image_paths = []
    cluster_labels = []
    hover_texts = []
    colors = []
    
    # Color palette for clusters
    color_palette = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]
    
    # Process images from the new JSON structure
    for img_data in data['images']:
        x_coords.append(img_data['coordinates']['x'])
        y_coords.append(img_data['coordinates']['y'])
        image_paths.append(img_data['path'])
        cluster_name = img_data['cluster_name']
        cluster_labels.append(cluster_name)
        
        # Create hover text
        filename = img_data['filename']
        hover_text = f"<b>{filename}</b><br>Cluster: {cluster_name}"
        hover_texts.append(hover_text)
        
        # Assign color based on cluster
        if cluster_name == 'noise':
            colors.append('#808080')  # Gray for noise
        else:
            cluster_id = img_data['cluster']
            colors.append(color_palette[cluster_id % len(color_palette)])
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Group by cluster for legend
    unique_clusters = sorted(set(cluster_labels), key=lambda x: (x != 'noise', x))
    
    for cluster in unique_clusters:
        # Mask is a list of global indices for this specific cluster
        mask = [i for i, c in enumerate(cluster_labels) if c == cluster]
        
        fig.add_trace(go.Scatter(
            x=[x_coords[i] for i in mask],
            y=[y_coords[i] for i in mask],
            mode='markers',
            name=cluster,
            marker=dict(
                size=12,
                color=colors[mask[0]],
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=[hover_texts[i] for i in mask],
            hoverinfo='text',
            # Pass the global index (mask) as customdata to identify the image uniquely
            customdata=mask 
        ))
    
    fig.update_layout(
        title="Image Clusters in 2D Embedding Space",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig, image_paths, x_coords, y_coords, cluster_labels

def display_image_grid(data, selected_cluster=None):
    """Display images in a grid layout."""
    
    st.subheader("Image Gallery")
    
    # Filter by cluster if selected
    if selected_cluster and selected_cluster != "All":
        images_to_show = [img for img in data['images'] if img['cluster_name'] == selected_cluster]
        st.write(f"Showing images from: **{selected_cluster}** ({len(images_to_show)} images)")
    else:
        images_to_show = data['images']
        st.write(f"Showing all images ({len(images_to_show)} images)")
    
    # Create columns for grid layout
    cols_per_row = 4
    
    for i in range(0, len(images_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(images_to_show):
                img_data = images_to_show[i + j]
                img_path = img_data['path']
                cluster_name = img_data['cluster_name']
                
                try:
                    image = Image.open(img_path)
                    
                    with col:
                        st.image(image, use_container_width=True)
                        st.caption(f"📁 {img_data['filename']}")
                        st.caption(f"🏷️ {cluster_name}")
                except Exception as e:
                    with col:
                        st.error(f"Error loading image: {img_data['filename']}")

# Sidebar
st.sidebar.header("Settings")

json_file = st.sidebar.text_input(
    "JSON File Path",
    value="image_clusters_visualization.json",
    help="Path to the cluster visualization JSON file"
)

# Load data
data = load_cluster_data(json_file)

if data is not None:
    # Display statistics
    st.sidebar.subheader("Cluster Statistics")
    total_images = len(data['images'])
    
    # Count clusters and noise
    cluster_names = set(img['cluster_name'] for img in data['images'])
    n_clusters = len([c for c in cluster_names if c != 'noise'])
    n_noise = len([img for img in data['images'] if img['cluster_name'] == 'noise'])
    
    st.sidebar.metric("Total Images", total_images)
    st.sidebar.metric("Number of Clusters", n_clusters)
    st.sidebar.metric("Noise/Outliers", n_noise)
    
    # Cluster selector
    st.sidebar.subheader("Filter by Cluster")
    cluster_options = ["All"] + sorted(cluster_names, key=lambda x: (x != 'noise', x))
    selected_cluster = st.sidebar.selectbox("Select Cluster", cluster_options)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🖼️ Image Gallery", "📋 Cluster Details"])
    
    with tab1:
        st.subheader("Interactive 2D Visualization")
        st.markdown("*Hover over points to see image details. Click on legend items to toggle clusters.*")
        
        fig, image_paths, x_coords, y_coords, cluster_labels = create_scatter_plot(data)
        
        # Display the plot with click events
        selected_points = st.plotly_chart(fig, use_container_width=True, key="scatter", on_select="rerun")
        
        # Display selected image when a point is clicked
        if selected_points and selected_points.selection and selected_points.selection["points"]:
            st.subheader("Selected Image")
            
            # Get the global indices passed via customdata
            selected_indices = [p["customdata"] for p in selected_points.selection["points"]]
            
            if selected_indices:
                # Find the corresponding image data using the global index
                selected_images = []
                for idx in selected_indices:
                    # Ensure index is within bounds
                    if isinstance(idx, int) and 0 <= idx < len(data['images']):
                        selected_images.append(data['images'][idx])
                
                # Display selected images in columns
                cols = st.columns(min(3, len(selected_images)))
                for i, img_data in enumerate(selected_images[:3]):
                    try:
                        image = Image.open(img_data['path'])
                        with cols[i]:
                            st.image(image, use_container_width=True)
                            st.write(f"**{img_data['filename']}**")
                            st.write(f"Cluster: {img_data['cluster_name']}")
                            st.write(f"Position: ({img_data['coordinates']['x']:.3f}, {img_data['coordinates']['y']:.3f})")
                    except Exception as e:
                        with cols[i]:
                            st.error(f"Error loading {img_data['filename']}")
        else:
            st.info("👆 Click on any point in the scatter plot above to view the corresponding image.")
    
    with tab2:
        display_image_grid(data, selected_cluster)
    
    with tab3:
        st.subheader("Cluster Details")
        
        # Group images by cluster
        clusters_dict = {}
        for img in data['images']:
            cluster_name = img['cluster_name']
            if cluster_name not in clusters_dict:
                clusters_dict[cluster_name] = []
            clusters_dict[cluster_name].append(img)
        
        for cluster_name in sorted(clusters_dict.keys(), key=lambda x: (x != 'noise', x)):
            with st.expander(f"{cluster_name} ({len(clusters_dict[cluster_name])} images)"):
                
                # Display cluster statistics
                images = clusters_dict[cluster_name]
                
                st.write("**Images in this cluster:**")
                for img_data in images:
                    st.write(f"- {img_data['filename']}")
                    st.write(f"  - Position: ({img_data['coordinates']['x']:.2f}, {img_data['coordinates']['y']:.2f})")
                
                # Display a preview of images in this cluster
                st.write("**Preview:**")
                preview_cols = st.columns(min(4, len(images)))
                for idx, img_data in enumerate(images[:4]):
                    try:
                        image = Image.open(img_data['path'])
                        with preview_cols[idx]:
                            st.image(image, use_container_width=True)
                    except Exception as e:
                        with preview_cols[idx]:
                            st.error("Error loading")
else:
    st.warning("Please provide a valid JSON file path in the sidebar.")
    st.info("""
    To generate the visualization data, run:
    ```bash
    python analysis/clip.py
    ```
    This will create the `image_clusters_visualization.json` file.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit and CLIP")