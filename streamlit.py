# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Polygon
import json
import os
import tempfile
import zipfile
from io import BytesIO
import base64
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import requests
import time
import math
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.mask import mask
from shapely.geometry import mapping
from collections import Counter

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🗺️ Interactive Polygon Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Model Architecture (same as your training)
# ========================
class ImageClassifier(nn.Module):
    def __init__(self, architecture, num_classes, pretrained=False):
        super(ImageClassifier, self).__init__()

        if architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        elif architecture == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            self.backbone.classifier = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

        elif architecture == "vit_b_16":
            self.backbone = models.vit_b_16(pretrained=pretrained)
            self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, num_classes)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x):
        return self.backbone(x)

# ========================
# Model Loading Functions
# ========================
@st.cache_resource(show_spinner=True)
def download_and_load_model():
    """Download ONNX model from Hugging Face Hub and load it"""
    try:
        from huggingface_hub import hf_hub_download
        
        # Hugging Face repository details
        repo_id = "woov/resnet50_landuse_5_classification"
        model_filename = "agga-v2.onnx"
        info_filename = "agga-v2_info.json"
        
        # Create info container
        info_container = st.container()
        with info_container:
            with st.expander("🤖 Model Loading Details", expanded=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()
                
                # Step 1: Download model
                status_text.info("📥 Downloading ONNX model from Hugging Face Hub...")
                details_text.text(f"Repository: {repo_id}")
                progress_bar.progress(10)
                
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename,
                    cache_dir=None
                )
                progress_bar.progress(40)
                status_text.success("✅ Model downloaded successfully!")
                
                # Step 2: Download metadata
                info_path = None
                try:
                    status_text.info("📄 Downloading model metadata...")
                    progress_bar.progress(50)
                    info_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=info_filename,
                        cache_dir=None
                    )
                    progress_bar.progress(60)
                    status_text.success("✅ Model metadata downloaded!")
                except Exception as e:
                    status_text.warning("⚠️ Model metadata not found, using defaults")
                    info_path = None
                
                # Step 3: Load ONNX model
                status_text.info("🔧 Loading ONNX model...")
                progress_bar.progress(70)
                
                import onnxruntime as ort
                ort_session = ort.InferenceSession(model_path)
                progress_bar.progress(80)
                
                # Step 4: Load configuration
                config = {}
                class_to_idx = {}
                idx_to_class = {}
                
                if info_path and os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                        config = {
                            'architecture': model_info.get('architecture', 'resnet50'),
                            'num_classes': model_info.get('num_classes', 5),
                            'image_size': model_info.get('image_size', 640)
                        }
                        class_to_idx = model_info.get('class_to_idx', {})
                        idx_to_class = {int(k): v for k, v in model_info.get('idx_to_class', {}).items()}
                else:
                    # Default config for your ResNet50 land use classification model
                    config = {
                        'architecture': 'resnet50',
                        'num_classes': 5,
                        'image_size': 640
                    }
                    
                    # Try to infer from ONNX model output
                    output_shape = ort_session.get_outputs()[0].shape
                    if len(output_shape) >= 2:
                        num_classes = output_shape[-1] if output_shape[-1] != -1 else output_shape[1]
                        config['num_classes'] = num_classes
                        
                    # Create default class mappings for land use classification
                    default_classes = ['urban', 'agriculture', 'forest', 'water', 'other'][:config['num_classes']]
                    class_to_idx = {cls: i for i, cls in enumerate(default_classes)}
                    idx_to_class = {i: cls for i, cls in enumerate(default_classes)}
                
                progress_bar.progress(90)
                
                # Final status
                progress_bar.progress(100)
                status_text.success("🎉 Model loaded successfully!")
                
                # Show final details
                details_text.markdown(f"""
                **📊 Model Details:**
                - 🏗️ Architecture: {config['architecture']}
                - 📊 Classes: {len(class_to_idx)} → {list(class_to_idx.keys())}
                - 🖼️ Input size: {config['image_size']}×{config['image_size']}
                - 💾 Cached at: `{model_path}`
                """)
        
        return ort_session, config, class_to_idx, idx_to_class
        
    except Exception as e:
        st.error(f"❌ Failed to load ONNX model from Hugging Face: {str(e)}")
        st.error("Please check your internet connection or model repository")
        st.stop()

@st.cache_data
def get_prediction_transforms(image_size=640):
    """Get image transforms for prediction"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ========================
# Map Functions (same as your code)
# ========================
def lonlat_to_tile(lon: float, lat: float, zoom: int) -> (int, int):
    """Convert WGS84 lon/lat to Google tile indices at given zoom"""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    xtile = max(0, min(xtile, n - 1))
    ytile = max(0, min(ytile, n - 1))
    return xtile, ytile

def tile_to_lonlat(x: int, y: int, zoom: int) -> (float, float):
    """Convert tile indices back to lon/lat coordinates"""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat

def download_tile(url, retries=3, delay=1):
    """Download a single map tile with retries"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except requests.RequestException:
            time.sleep(delay)
    return None

def polygon_to_image(poly, zoom: int, scale: int = 2):
    """Convert polygon to satellite image - EXACTLY same as your code"""
    if poly is None or poly.is_empty or not poly.is_valid:
        return None

    minx, miny, maxx, maxy = poly.bounds
    if any(math.isnan(v) for v in (minx, miny, maxx, maxy)):
        return None

    x_min, y_min = lonlat_to_tile(minx, maxy, zoom)
    x_max, y_max = lonlat_to_tile(maxx, miny, zoom)

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1

    tile_size = 256 * scale
    mosaic = Image.new("RGB", (cols * tile_size, rows * tile_size))

    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&scale={scale}"

    for ix, x in enumerate(range(x_min, x_max + 1)):
        for iy, y in enumerate(range(y_min, y_max + 1)):
            url = url_template.format(x=x, y=y, z=zoom, scale=scale)
            tile = download_tile(url)
            if tile is None:
                return None

            if scale != 1:
                tile = tile.resize((tile_size, tile_size), Image.LANCZOS)

            mosaic.paste(tile, (ix * tile_size, iy * tile_size))

    actual_minx, actual_maxy = tile_to_lonlat(x_min, y_min, zoom)
    actual_maxx, actual_miny = tile_to_lonlat(x_max + 1, y_max + 1, zoom)

    transform = from_bounds(actual_minx, actual_miny, actual_maxx, actual_maxy,
                           mosaic.width, mosaic.height)

    arr = np.array(mosaic)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=3,
            dtype=arr.dtype, crs="EPSG:4326", transform=transform
        ) as tmp:
            for b in range(3):
                tmp.write(arr[:, :, b], b + 1)

        with memfile.open() as src:
            out_image, _ = mask(src, [mapping(poly)], crop=False)

    if not out_image.any():
        return None

    png_arr = np.transpose(out_image, (1, 2, 0))
    img = Image.fromarray(png_arr)

    from PIL import ImageOps
    img = ImageOps.pad(img, (640, 640), color=(0, 0, 0))

    return img

# ========================
# Prediction Functions
# ========================
def predict_single_image(ort_session, image, transform, idx_to_class):
    """Predict single image using ONNX Runtime"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Apply transforms
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).numpy()  # Convert to numpy for ONNX

    # Run ONNX inference
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_tensor})
    
    # Process outputs
    logits = outputs[0][0]  # Get first batch item
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx]

    predicted_class = idx_to_class.get(predicted_class_idx, f'class_{predicted_class_idx}')
    return predicted_class, float(confidence), probabilities

# ========================
# UI Functions
# ========================
def create_map(center_lat=-7.25, center_lon=112.75, zoom_start=12):
    """Create interactive map with drawing tools"""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add drawing functionality using plugins
    from folium.plugins import Draw
    draw = Draw(
        export=True,
        filename='polygons.geojson',
        position='topleft',
        draw_options={
            'polygon': {
                'allowIntersection': False,
                'showArea': True,
                'drawError': {
                    'color': '#e1e100',
                    'message': 'Polygon intersects with existing one!'
                },
                'shapeOptions': {
                    'color': '#ff0000',
                    'weight': 3,
                    'fillOpacity': 0.3
                }
            },
            'rectangle': {
                'shapeOptions': {
                    'color': '#0000ff',
                    'weight': 3,
                    'fillOpacity': 0.3
                }
            },
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polyline': False
        },
        edit_options={
            'poly': {
                'allowIntersection': False
            }
        }
    )
    draw.add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def process_drawn_features(map_data):
    """Process features drawn on the map"""
    if map_data['all_drawings'] is None:
        return None
    
    features = []
    for drawing in map_data['all_drawings']:
        if drawing['geometry']['type'] in ['Polygon', 'Rectangle']:
            features.append({
                'type': 'Feature',
                'geometry': drawing['geometry'],
                'properties': drawing.get('properties', {})
            })
    
    if not features:
        return None
    
    # Convert to GeoDataFrame
    polygons = []
    for i, feature in enumerate(features):
        geom = shape(feature['geometry'])
        polygons.append({
            'id': i,
            'geometry': geom,
            'area': geom.area,
            'perimeter': geom.length
        })
    
    gdf = gpd.GeoDataFrame(polygons, crs='EPSG:4326')
    return gdf

# ========================
# Main Application
# ========================
def main():
    st.title("🗺️ Automated Grid-based Geo Annotator")
    st.markdown("Draw polygons on the map to classify satellite imagery using deep learning")

    # Sidebar
    with st.sidebar:
        st.header("🗺️ AGGA-v2")
        st.markdown("### 📋 Description")
        st.markdown("""
        5 Classification : 
        - slum+slum-to-normal
        - normal+normal-premium
        - premium
        - industri
        - komersial
        """)
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Draw polygons** on the map using drawing tools
        2. **Click 'Predict'** to classify drawn areas
        3. **View results** in the results section
        """)
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.markdown("**Image Capture Settings:**")
        st.markdown("- Zoom Level: 17 (fixed)")
        st.markdown("- Scale: 2x (fixed)")
        st.markdown("- Image Size: 640×640 (fixed)")
        st.markdown("*Settings are fixed to match training data*")

    # Load model
    if 'model_loaded' not in st.session_state:
        with st.spinner("Loading ONNX model..."):
            ort_session, config, class_to_idx, idx_to_class = download_and_load_model()
            transform = get_prediction_transforms(config['image_size'])
            
            st.session_state.ort_session = ort_session
            st.session_state.config = config
            st.session_state.class_to_idx = class_to_idx
            st.session_state.idx_to_class = idx_to_class
            st.session_state.transform = transform
            st.session_state.model_loaded = True

    # Main content
    st.subheader("🗺️ Interactive Map")
    
    # Fixed settings (same as training data)
    center_lat, center_lon, zoom_start = -7.25, 112.75, 12
    img_zoom, img_scale = 17, 2  # Fixed to match training
    
    # Create and display map
    m = create_map(center_lat, center_lon, zoom_start)
    map_data = st_folium(
        m, 
        width=None,  # Use full width
        height=600,  # Increased height
        returned_objects=["all_drawings"]
    )
    
    # Process drawn polygons
    if map_data['all_drawings']:
        gdf = process_drawn_features(map_data)
        
        if gdf is not None and len(gdf) > 0:
            st.success(f"✅ {len(gdf)} polygon(s) drawn")
            
            # Show polygon info
            with st.expander("📊 Polygon Details"):
                for idx, row in gdf.iterrows():
                    st.write(f"**Polygon {idx+1}:**")
                    st.write(f"- Area: {row.geometry.area:.8f}°²")
                    st.write(f"- Perimeter: {row.geometry.length:.8f}°")
                    bounds = row.geometry.bounds
                    st.write(f"- Bounds: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")
            
            # Prediction button
            if st.button("🤖 Predict Classifications", type="primary", use_container_width=True):
                predict_polygons(gdf, img_zoom, img_scale)
        else:
            st.info("👆 Draw some polygons on the map to get started")
    else:
        st.info("👆 Draw some polygons on the map to get started")

    # Results section - moved below map
    st.markdown("---")  # Add separator
    st.subheader("📊 Results")
    
    if 'predictions' in st.session_state and st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
        display_prediction_results()
    else:
        st.info("Draw polygons and click 'Predict' to see results here")

def predict_polygons(gdf, zoom, scale):
    """Predict classifications for drawn polygons using ONNX"""
    
    # Get model components from session state
    ort_session = st.session_state.ort_session
    transform = st.session_state.transform
    idx_to_class = st.session_state.idx_to_class
    
    st.subheader("🔄 Processing Polygons")
    
    predictions = []
    confidences = []
    all_probs = []
    captured_images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in gdf.iterrows():
        progress = (idx + 1) / len(gdf)
        progress_bar.progress(progress)
        status_text.text(f"Processing polygon {idx+1}/{len(gdf)}")
        
        try:
            # Generate image from polygon
            img = polygon_to_image(row.geometry, zoom, scale)
            
            if img is not None:
                # Make prediction using ONNX
                pred_class, confidence, probs = predict_single_image(
                    ort_session, img, transform, idx_to_class
                )
                
                predictions.append(pred_class)
                confidences.append(confidence)
                all_probs.append(probs)
                captured_images.append(img)
                
                st.success(f"Polygon {idx+1}: {pred_class} ({confidence:.1%})")
                
            else:
                predictions.append("unknown")
                confidences.append(0.0)
                all_probs.append(np.zeros(len(idx_to_class)))
                captured_images.append(None)
                st.error(f"Failed to capture image for polygon {idx+1}")
                
        except Exception as e:
            predictions.append("unknown")
            confidences.append(0.0)
            all_probs.append(np.zeros(len(idx_to_class)))
            captured_images.append(None)
            st.error(f"Error processing polygon {idx+1}: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Store results in session state
    result_gdf = gdf.copy()
    result_gdf['prediction'] = predictions
    result_gdf['confidence'] = confidences
    
    # Add probability columns
    for class_idx, class_name in idx_to_class.items():
        result_gdf[f'prob_{class_name}'] = [float(p[class_idx]) if len(p) > class_idx else 0.0 for p in all_probs]
    
    st.session_state.predictions = result_gdf
    st.session_state.captured_images = captured_images
    
    # Display summary
    st.subheader("📋 Prediction Summary")
    successful_preds = sum(1 for p in predictions if p != "unknown")
    failed_preds = len(predictions) - successful_preds
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Polygons", len(predictions))
    with col2:
        st.metric("Successful", successful_preds)
    with col3:
        st.metric("Failed", failed_preds)
    
    # Prediction distribution
    pred_counts = Counter(predictions)
    st.write("**Prediction Distribution:**")
    for pred, count in pred_counts.items():
        st.write(f"- {pred}: {count}")

def display_prediction_results():
    """Display prediction results"""
    if 'predictions' not in st.session_state:
        return
    
    result_gdf = st.session_state.predictions
    captured_images = st.session_state.get('captured_images', [])
    
    # Results table
    st.write("**📊 Detailed Results:**")
    display_cols = ['prediction', 'confidence']
    if len(result_gdf.columns) > 10:  # If has probability columns
        prob_cols = [col for col in result_gdf.columns if col.startswith('prob_')]
        display_cols.extend(prob_cols[:3])  # Show first 3 probability columns
    
    st.dataframe(result_gdf[display_cols], use_container_width=True)
    
    # Display captured images
    if captured_images and any(img is not None for img in captured_images):
        st.write("**🖼️ Captured Images:**")
        
        for idx, (img, pred, conf) in enumerate(zip(captured_images, result_gdf['prediction'], result_gdf['confidence'])):
            if img is not None:
                st.write(f"**Polygon {idx+1}:** {pred} ({conf:.1%} confidence)")
                st.image(img, width=200)
    
    # Download results with predictions
    st.write("**💾 Download Results:**")
    
    # Create download data
    download_gdf = result_gdf.copy()
    
    # Ensure all data is serializable for GeoJSON
    for col in download_gdf.columns:
        if col != 'geometry':
            if download_gdf[col].dtype == 'object':
                download_gdf[col] = download_gdf[col].astype(str)
            elif 'float' in str(download_gdf[col].dtype):
                download_gdf[col] = download_gdf[col].round(6)  # Round to 6 decimal places
    
    # Convert to GeoJSON
    geojson_data = download_gdf.to_json()
    
    st.download_button(
        label="🗺️ Download GeoJSON with Predictions",
        data=geojson_data,
        file_name="polygon_predictions.geojson",
        mime="application/json",
        help="Download polygons with classification results as GeoJSON"
    )

if __name__ == "__main__":
    main()