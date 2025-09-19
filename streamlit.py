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
# Model Configurations
# ========================
MODEL_CONFIGS = {
    "agga-v2": {
        "repo_id": "woov/resnet50_landuse_5_classification",
        "model_filename": "agga-v2.onnx",
        "info_filename": "agga-v2_info.json",
        "display_name": "AGGA-v2 (5 Classes)",
        "description": "5 Classification: slum+slum-to-normal, normal+normal-premium, premium, industri, komersial",
        "num_classes": 5,
        "default_classes": ['slum+slum-to-normal', 'normal+normal-premium', 'premium', 'industri', 'komersial']
    },
    "agga-v4": {
        "repo_id": "woov/resnet50_landuse_5_classification",
        "model_filename": "agga-v4.onnx",
        "info_filename": "agga-v4_info.json",
        "display_name": "AGGA-v4 (6 Classes)",
        "description": "6 Classification with detailed explanations",
        "num_classes": 6,
        "default_classes": ['green', 'slum+slum-to-normal', 'normal+normal-premium', 'premium', 'industri', 'komersial'],
        "class_labels": {
            1: "green",
            2: "slum+slum-to-normal", 
            3: "normal+normal-premium",
            4: "premium",
            5: "industri",
            6: "komersial"
        }
    }
}

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
def download_and_load_model(model_key):
    """Download ONNX model from Hugging Face Hub and load it"""
    try:
        from huggingface_hub import hf_hub_download
        
        # Get model configuration
        model_config = MODEL_CONFIGS[model_key]
        repo_id = model_config["repo_id"]
        model_filename = model_config["model_filename"]
        info_filename = model_config["info_filename"]
        
        # Create info container
        info_container = st.container()
        with info_container:
            with st.expander(f"🤖 Loading {model_config['display_name']}", expanded=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_text = st.empty()
                
                # Step 1: Download model
                status_text.info(f"📥 Downloading {model_config['display_name']} from Hugging Face Hub...")
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
                            'num_classes': model_info.get('num_classes', model_config['num_classes']),
                            'image_size': model_info.get('image_size', 640)
                        }
                        class_to_idx = model_info.get('class_to_idx', {})
                        idx_to_class = {int(k): v for k, v in model_info.get('idx_to_class', {}).items()}
                else:
                    # Default config
                    config = {
                        'architecture': 'resnet50',
                        'num_classes': model_config['num_classes'],
                        'image_size': 640
                    }
                    
                    # Try to infer from ONNX model output
                    output_shape = ort_session.get_outputs()[0].shape
                    if len(output_shape) >= 2:
                        num_classes = output_shape[-1] if output_shape[-1] != -1 else output_shape[1]
                        config['num_classes'] = num_classes
                        
                    # Create default class mappings
                    default_classes = model_config['default_classes'][:config['num_classes']]
                    
                    # For AGGA-v4, handle the 1-6 indexing
                    if model_key == "agga-v4":
                        # Map 0-based indices to 1-6 classes with explanations
                        class_explanations = model_config['class_explanations']
                        class_to_idx = {class_explanations[i+1]: i for i in range(len(default_classes))}
                        idx_to_class = {i: class_explanations[i+1] for i in range(len(default_classes))}
                    else:
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
                - 📊 Classes: {len(class_to_idx)}
                - 🖼️ Input size: {config['image_size']}×{config['image_size']}
                - 💾 Cached at: `{model_path}`
                """)
                
                # Show class details for both models
                if model_key == "agga-v4":
                    st.markdown("**🏷️ Class Details:**")
                    class_labels = model_config['class_labels']
                    for class_idx, label in class_labels.items():
                        st.markdown(f"- **{class_idx}**: {label}")
                else:
                    st.markdown("**🏷️ Classes:**")
                    classes = list(class_to_idx.keys())
                    for i, cls in enumerate(classes):
                        st.markdown(f"- **{i}**: {cls}")
        
        return ort_session, config, class_to_idx, idx_to_class, model_key
        
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
    """Convert polygon to satellite image - Using the EXACT method from extract_patches.py"""
    if poly is None or poly.is_empty or not poly.is_valid:
        return None

    minx, miny, maxx, maxy = poly.bounds
    if any(math.isnan(v) for v in (minx, miny, maxx, maxy)):
        return None

    # FIXED: Correct tile range calculation (same as extract_patches.py)
    # Top-left corner of bounding box
    x_min, y_min = lonlat_to_tile(minx, maxy, zoom)
    # Bottom-right corner of bounding box
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

    # FIXED: Calculate actual geographic bounds covered by the tiles
    actual_minx, actual_maxy = tile_to_lonlat(x_min, y_min, zoom)
    actual_maxx, actual_miny = tile_to_lonlat(x_max + 1, y_max + 1, zoom)

    # Create proper geotransform using actual tile bounds
    transform = from_bounds(actual_minx, actual_miny, actual_maxx, actual_maxy,
                           mosaic.width, mosaic.height)

    # Convert mosaic to numpy array
    arr = np.array(mosaic)

    # Apply polygon mask using rasterio
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=3,
            dtype=arr.dtype,
            crs="EPSG:4326",
            transform=transform
        ) as tmp:
            # Write RGB bands
            for b in range(3):
                tmp.write(arr[:, :, b], b + 1)

        # Read back and apply mask
        with memfile.open() as src:
            out_image, _ = mask(src, [mapping(poly)], crop=False)

    # Check if mask produced valid output
    if not out_image.any():
        return None

    # Convert to H×W×3 format and create PIL image
    png_arr = np.transpose(out_image, (1, 2, 0))
    img = Image.fromarray(png_arr)

    # Pad to 640×640 (same as extract_patches.py)
    from PIL import ImageOps
    img = ImageOps.pad(img, (640, 640), color=(0, 0, 0))

    return img

# ========================
# Prediction Functions
# ========================
def predict_single_image(ort_session, image, transform, idx_to_class, model_key=None):
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

    # For AGGA-v4, we need to map the raw prediction to the label format
    if model_key == "agga-v4":
        # The model outputs 0,1,2,3,4,5 but we want to display as "1 (green)", "2 (slum+slum-to-normal)", etc.
        class_labels = MODEL_CONFIGS["agga-v4"]["class_labels"]
        actual_class_num = predicted_class_idx + 1  # Convert 0-based to 1-based
        class_label = class_labels.get(actual_class_num, f'class_{actual_class_num}')
        predicted_class = f"{actual_class_num} ({class_label})"
    else:
        predicted_class = idx_to_class.get(predicted_class_idx, f'class_{predicted_class_idx}')
    
    return predicted_class, float(confidence), probabilities

# ========================
# Geocoding Service
# ========================
class GeocodeService:
    """Handle geocoding using Google Maps Geocoding API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_address(self, address: str) -> tuple:
        """Geocode an address to get latitude and longitude"""
        try:
            params = {
                'address': address,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                location = result['geometry']['location']
                formatted_address = result['formatted_address']
                
                return (
                    location['lat'],
                    location['lng'],
                    formatted_address
                )
            else:
                return None, None, None
                
        except Exception as e:
            st.error(f"Geocoding error: {str(e)}")
            return None, None, None

def initialize_geocode_service():
    """Initialize geocoding service with API key from secrets"""
    try:
        google_api_key = st.secrets["google"]["api_key"]
        if 'geocode_service' not in st.session_state:
            st.session_state.geocode_service = GeocodeService(google_api_key)
        return st.session_state.geocode_service
    except KeyError:
        st.warning("Google Maps API key not found. Location search features unavailable.")
        st.session_state.geocode_service = None
        return None

# ========================
# Voting Prediction Functions
# ========================
def meters_to_degrees(meters, latitude):
    """Convert meters to degrees at a given latitude"""
    import math
    # Earth's radius in meters
    earth_radius = 6378137.0
    
    # Convert latitude to radians
    lat_rad = math.radians(latitude)
    
    # Degrees per meter for longitude (varies with latitude)
    degrees_per_meter_lon = 1 / (earth_radius * math.cos(lat_rad) * math.pi / 180)
    
    # Degrees per meter for latitude (constant)
    degrees_per_meter_lat = 1 / (earth_radius * math.pi / 180)
    
    return meters * degrees_per_meter_lon, meters * degrees_per_meter_lat

def create_grid_points(center_lat, center_lon, distance_meters):
    """Create 3x3 grid of points around center with specified spacing"""
    # Convert distance to degrees
    lon_degrees, lat_degrees = meters_to_degrees(distance_meters, center_lat)
    
    # Create 3x3 grid (9 points)
    grid_points = []
    grid_positions = []
    
    # Grid positions: from -1 to +1 in both directions
    for i in range(-1, 2):  # -1, 0, 1 (rows: top, center, bottom)
        for j in range(-1, 2):  # -1, 0, 1 (cols: left, center, right)
            lat = center_lat + (i * lat_degrees)
            lon = center_lon + (j * lon_degrees)
            grid_points.append((lat, lon))
            
            # Position labels for display
            row_label = ["Top", "Center", "Bottom"][i + 1]
            col_label = ["Left", "Center", "Right"][j + 1]
            grid_positions.append(f"{row_label}-{col_label}")
    
    return grid_points, grid_positions

def point_to_bbox(lat, lon, buffer_meters):
    """Convert a point to a bounding box with specified buffer in meters"""
    from shapely.geometry import box
    
    # Convert buffer meters to degrees
    lon_degrees, lat_degrees = meters_to_degrees(buffer_meters, lat)
    
    # Create bounding box
    minx = lon - lon_degrees
    maxx = lon + lon_degrees
    miny = lat - lat_degrees
    maxy = lat + lat_degrees
    
    return box(minx, miny, maxx, maxy)

def parse_coordinates(coord_string):
    """Parse coordinate string in various formats"""
    try:
        # Remove extra whitespace and common separators
        coord_string = coord_string.strip().replace(',', ' ').replace(';', ' ')
        
        # Split by spaces and filter empty strings
        parts = [p.strip() for p in coord_string.split() if p.strip()]
        
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            
            # Basic validation
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon, None
            else:
                return None, None, "Coordinates out of valid range (lat: -90 to 90, lon: -180 to 180)"
        else:
            return None, None, "Please provide both latitude and longitude"
            
    except ValueError:
        return None, None, "Invalid coordinate format. Use format like: -7.25 112.75"

def predict_voting_grid(center_lat, center_lon, distance_meters, model_components, zoom=17, scale=2):
    """Predict using voting system on 3x3 grid"""
    
    # Get model components
    ort_session, transform, idx_to_class, current_model = model_components
    
    # Create grid points
    grid_points, grid_positions = create_grid_points(center_lat, center_lon, distance_meters)
    
    # Calculate bounding box size (half of distance for radius)
    bbox_buffer = distance_meters / 2
    
    st.info(f"Creating 3x3 grid with {distance_meters}m spacing, {distance_meters}x{distance_meters}m bounding boxes")
    
    predictions = []
    confidences = []
    all_probs = []
    successful_points = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ((lat, lon), position) in enumerate(zip(grid_points, grid_positions)):
        progress = (idx + 1) / len(grid_points)
        progress_bar.progress(progress)
        status_text.text(f"Processing {position} point ({idx+1}/9)")
        
        try:
            # Create bounding box around point
            bbox_poly = point_to_bbox(lat, lon, bbox_buffer)
            
            # Generate image from bounding box
            img = polygon_to_image(bbox_poly, zoom, scale)
            
            if img is not None:
                # Make prediction
                pred_class, confidence, probs = predict_single_image(
                    ort_session, img, transform, idx_to_class, current_model
                )
                
                predictions.append(pred_class)
                confidences.append(confidence)
                all_probs.append(probs)
                successful_points.append(position)
                
                st.success(f"{position}: {pred_class} ({confidence:.1%})")
                
            else:
                st.error(f"Failed to capture image for {position}")
                predictions.append("unknown")
                confidences.append(0.0)
                all_probs.append(np.zeros(len(idx_to_class)))
                
        except Exception as e:
            st.error(f"Error processing {position}: {str(e)}")
            predictions.append("unknown")
            confidences.append(0.0)
            all_probs.append(np.zeros(len(idx_to_class)))
    
    progress_bar.progress(1.0)
    status_text.text("✅ Grid processing complete!")
    
    return predictions, confidences, all_probs, grid_positions, grid_points

def calculate_voting_result(predictions, confidences, grid_positions):
    """Calculate voting result from grid predictions"""
    from collections import Counter
    
    # Filter out unknown predictions for voting
    valid_predictions = [p for p in predictions if p != "unknown"]
    
    if not valid_predictions:
        return "unknown", 0.0, {}, True, "All predictions failed"
    
    # Count votes
    vote_counts = Counter(valid_predictions)
    total_votes = len(valid_predictions)
    
    # Find winner(s)
    max_votes = max(vote_counts.values())
    winners = [pred for pred, count in vote_counts.items() if count == max_votes]
    
    # Check for tie
    is_tie = len(winners) > 1
    
    if is_tie:
        # In case of tie, pick the one with highest average confidence
        tie_confidences = {}
        for winner in winners:
            winner_indices = [i for i, p in enumerate(predictions) if p == winner]
            avg_confidence = np.mean([confidences[i] for i in winner_indices])
            tie_confidences[winner] = avg_confidence
        
        final_winner = max(tie_confidences.keys(), key=lambda x: tie_confidences[x])
        final_confidence = tie_confidences[final_winner]
        tie_message = f"Tie detected! {', '.join(winners)} each got {max_votes} votes. Winner selected by highest confidence."
    else:
        final_winner = winners[0]
        winner_indices = [i for i, p in enumerate(predictions) if p == final_winner]
        final_confidence = np.mean([confidences[i] for i in winner_indices])
        tie_message = None
    
    # Create vote breakdown
    vote_breakdown = {}
    for pred, count in vote_counts.items():
        percentage = (count / total_votes) * 100
        vote_breakdown[pred] = {"votes": count, "percentage": percentage}
    
    return final_winner, final_confidence, vote_breakdown, is_tie, tie_message

def create_voting_results_map(center_lat, center_lon, distance_meters, grid_points, grid_positions, predictions, final_prediction):
    """Create map showing voting grid results"""
    
    # Create map centered on the selected point
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color mapping for predictions
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'brown', 'cyan', 'magenta']
    
    unique_predictions = list(set(predictions))
    color_map = {pred: colors[i % len(colors)] for i, pred in enumerate(unique_predictions)}
    
    # Add center point marker (user's selected point)
    folium.Marker(
        [center_lat, center_lon],
        popup=f"<b>🎯 Center Point</b><br>Grid: {distance_meters}m spacing<br>Final: {final_prediction}",
        tooltip=f"🎯 Center Point",
        icon=folium.Icon(color='red', icon='bullseye')
    ).add_to(m)
    
    # Add grid point markers
    for (lat, lon), position, prediction in zip(grid_points, grid_positions, predictions):
        color = color_map.get(prediction, 'gray')
        
        popup_text = f"""
        <div style="font-family: Arial, sans-serif;">
            <b>{position}</b><br>
            <b>Prediction:</b> {prediction}<br>
            <b>Coordinates:</b> {lat:.6f}, {lon:.6f}
        </div>
        """
        
        # Use different icon for center vs grid points
        if position == "Center-Center":
            icon_name = 'star'
            icon_color = 'red'
        else:
            icon_name = 'circle'
            icon_color = 'blue' if prediction != 'unknown' else 'gray'
        
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{position}: {prediction}",
            icon=folium.Icon(color=icon_color, icon=icon_name)
        ).add_to(m)
    
    # Add proper 3x3 grid lines
    lat_spacing = meters_to_degrees(distance_meters, center_lat)[1]
    lon_spacing = meters_to_degrees(distance_meters, center_lat)[0]
    
    # Create grid boundary points
    top_lat = center_lat + lat_spacing
    bottom_lat = center_lat - lat_spacing
    left_lon = center_lon - lon_spacing
    right_lon = center_lon + lon_spacing
    
    # Draw horizontal grid lines (3 lines total)
    for i in range(-1, 2):  # Top, center, bottom
        lat = center_lat + (i * lat_spacing)
        folium.PolyLine(
            [[lat, left_lon], [lat, right_lon]], 
            color='yellow', weight=2, opacity=0.7
        ).add_to(m)
    
    # Draw vertical grid lines (3 lines total)
    for j in range(-1, 2):  # Left, center, right
        lon = center_lon + (j * lon_spacing)
        folium.PolyLine(
            [[top_lat, lon], [bottom_lat, lon]], 
            color='yellow', weight=2, opacity=0.7
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def voting_prediction_tab():
    """Tab for voting prediction with grid sampling"""
    st.subheader("🗳️ Voting Prediction")
    st.markdown("Set a point and create a 3x3 prediction grid with majority voting")
    
    # Initialize geocoding service
    geocode_service = initialize_geocode_service()
    
    # Point selection section
    st.markdown("### 📍 Set Center Point")
    
    coord_input = st.text_input(
        "Enter Coordinates (lat lon):",
        placeholder="-7.25 112.75",
        help="Enter coordinates in format: latitude longitude (e.g., -7.25 112.75)"
    )
    
    if coord_input.strip():
        lat, lon, error = parse_coordinates(coord_input)
        if error:
            st.error(error)
            center_point = None
        else:
            st.success(f"📍 Coordinates set: {lat:.6f}, {lon:.6f}")
            center_point = (lat, lon)
            st.session_state.voting_center = center_point
    else:
        center_point = st.session_state.get('voting_center', None)
    
    # Grid configuration
    st.markdown("### ⚙️ Grid Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        distance_options = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        selected_distance = st.selectbox(
            "Grid spacing & box size (meters):",
            distance_options,
            index=1,  # Default to 100m
            help="Distance between grid points AND size of bounding boxes"
        )
    
    with col2:
        st.markdown("**Grid Preview:**")
        st.markdown(f"- **Grid**: 3×3 = 9 prediction points")
        st.markdown(f"- **Spacing**: {selected_distance}m between points")
        st.markdown(f"- **Boxes**: {selected_distance}×{selected_distance}m per point")
        st.markdown(f"- **Total area**: ~{selected_distance*2}×{selected_distance*2}m")
    
    # Show current center point
    if center_point:
        lat, lon = center_point
        st.info(f"📍 Center point set: {lat:.6f}, {lon:.6f}")
        
        # Prediction button
        current_model_name = MODEL_CONFIGS[st.session_state.current_model]["display_name"]
        if st.button(f"🗳️ Run Voting Prediction using {current_model_name}", type="primary", use_container_width=True):
            
            # Get model components
            model_components = (
                st.session_state.ort_session,
                st.session_state.transform,
                st.session_state.idx_to_class,
                st.session_state.current_model
            )
            
            # Run voting prediction
            predictions, confidences, all_probs, grid_positions, grid_points = predict_voting_grid(
                lat, lon, selected_distance, model_components
            )
            
            # Calculate voting result
            final_prediction, final_confidence, vote_breakdown, is_tie, tie_message = calculate_voting_result(
                predictions, confidences, grid_positions
            )
            
            # Store results
            st.session_state.voting_results = {
                'center_point': center_point,
                'distance': selected_distance,
                'predictions': predictions,
                'confidences': confidences,
                'grid_positions': grid_positions,
                'grid_points': grid_points,
                'final_prediction': final_prediction,
                'final_confidence': final_confidence,
                'vote_breakdown': vote_breakdown,
                'is_tie': is_tie,
                'tie_message': tie_message,
                'model_used': st.session_state.current_model
            }
    else:
        st.warning("📍 Please set a center point using coordinates or search")
    
    # Display results
    if 'voting_results' in st.session_state:
        st.markdown("---")
        st.subheader("🗳️ Voting Results")
        
        results = st.session_state.voting_results
        
        # Final result summary
        st.markdown("### 🏆 Final Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Winner", results['final_prediction'])
        with col2:
            st.metric("Confidence", f"{results['final_confidence']:.1%}")
        with col3:
            model_name = MODEL_CONFIGS[results['model_used']]["display_name"]
            st.metric("Model", model_name)
        
        # Tie information
        if results['is_tie']:
            st.warning(f"⚠️ {results['tie_message']}")
        
        # Vote breakdown
        st.markdown("### 📊 Vote Breakdown")
        vote_df = []
        for pred, data in results['vote_breakdown'].items():
            vote_df.append({
                'Prediction': pred,
                'Votes': data['votes'],
                'Percentage': f"{data['percentage']:.1f}%"
            })
        
        if vote_df:
            st.dataframe(pd.DataFrame(vote_df), use_container_width=True)
        
        # Individual predictions
        st.markdown("### 📋 Individual Grid Predictions")
        grid_df = []
        for pos, pred, conf in zip(results['grid_positions'], results['predictions'], results['confidences']):
            grid_df.append({
                'Position': pos,
                'Prediction': pred,
                'Confidence': f"{conf:.1%}"
            })
        
        st.dataframe(pd.DataFrame(grid_df), use_container_width=True)
        
        # Results map
        st.markdown("### 🗺️ Grid Visualization")
        results_map = create_voting_results_map(
            results['center_point'][0], results['center_point'][1],
            results['distance'], results['grid_points'], results['grid_positions'],
            results['predictions'], results['final_prediction']
        )
        st_folium(results_map, width=None, height=600, key="voting_results_map")
        
        # Download results
        st.markdown("### 💾 Download Results")
        
        # Create summary data
        summary_data = {
            'center_coordinates': results['center_point'],
            'grid_spacing_meters': results['distance'],
            'final_prediction': results['final_prediction'],
            'final_confidence': results['final_confidence'],
            'is_tie': results['is_tie'],
            'tie_message': results['tie_message'],
            'model_used': results['model_used'],
            'vote_breakdown': results['vote_breakdown'],
            'grid_predictions': [
                {
                    'position': pos,
                    'coordinates': point,
                    'prediction': pred,
                    'confidence': conf
                }
                for pos, point, pred, conf in zip(
                    results['grid_positions'], 
                    results['grid_points'], 
                    results['predictions'], 
                    results['confidences']
                )
            ]
        }
        
        # Convert to JSON for download
        import json
        json_data = json.dumps(summary_data, indent=2)
        
        st.download_button(
            label="📄 Download Voting Results (JSON)",
            data=json_data,
            file_name=f"voting_prediction_{results['model_used']}.json",
            mime="application/json",
            help="Download detailed voting prediction results"
        )
def create_results_map(gdf_with_predictions):
    """Create a map showing predictions grouped by class"""
    if gdf_with_predictions.empty:
        return None
    
    # Calculate map center
    bounds = gdf_with_predictions.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color mapping for different predictions (avoid white/light colors)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'brown', 'cyan', 'magenta']
    
    # Get unique predictions and create color mapping
    unique_predictions = gdf_with_predictions['prediction'].unique()
    color_map = {pred: colors[i % len(colors)] for i, pred in enumerate(unique_predictions)}
    
    # Group polygons by prediction class and add as separate layers
    for prediction in unique_predictions:
        class_polygons = gdf_with_predictions[gdf_with_predictions['prediction'] == prediction]
        color = color_map[prediction]
        count = len(class_polygons)
        
        # Create feature group for this prediction class
        feature_group = folium.FeatureGroup(name=f"{prediction} ({count})", show=True)
        
        # Add all polygons of this class to the feature group
        for idx, row in class_polygons.iterrows():
            confidence = row.get('confidence', 0)
            
            # Create popup text
            popup_text = f"""
            <div style="font-family: Arial, sans-serif;">
                <b>Polygon {idx}</b><br>
                <b>Class:</b> {prediction}<br>
                <b>Confidence:</b> {confidence:.1%}
            </div>
            """
            
            # Add polygon to feature group
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.5,
                    'opacity': 0.8
                },
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{prediction} ({confidence:.1%})"
            ).add_to(feature_group)
        
        # Add feature group to map
        feature_group.add_to(m)
    
    # Add improved legend with better visibility
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: auto; 
                background-color: rgba(255, 255, 255, 0.9); 
                border: 2px solid #333; 
                border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                z-index: 9999; 
                font-family: Arial, sans-serif;
                font-size: 12px; 
                padding: 10px;">
    <div style="font-weight: bold; font-size: 14px; margin-bottom: 8px; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
        📊 Predictions
    </div>
    '''
    
    # Sort predictions by count for better display
    pred_counts = [(pred, len(gdf_with_predictions[gdf_with_predictions['prediction'] == pred])) 
                   for pred in unique_predictions]
    pred_counts.sort(key=lambda x: x[1], reverse=True)
    
    for pred, count in pred_counts:
        color = color_map[pred]
        percentage = (count / len(gdf_with_predictions)) * 100
        legend_html += f'''
        <div style="margin: 3px 0; display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: {color}; 
                        border: 1px solid #333; margin-right: 8px; flex-shrink: 0;"></div>
            <span style="color: #333; font-size: 11px; line-height: 1.2;">
                <b>{pred}</b><br>
                {count} polygons ({percentage:.1f}%)
            </span>
        </div>
        '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    return m

def process_geojson_batch(uploaded_file, model_components, zoom=17, scale=2):
    """Process uploaded GeoJSON file and return predictions"""
    try:
        # Read GeoJSON
        gdf = gpd.read_file(uploaded_file)
        gdf = gdf.to_crs('EPSG:4326')  # Ensure WGS84
        
        # Remove invalid geometries
        gdf = gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
        
        if len(gdf) == 0:
            st.error("No valid polygons found in the GeoJSON file")
            return None
        
        # Get model components
        ort_session, transform, idx_to_class, current_model = model_components
        
        st.info(f"Processing {len(gdf)} polygons from GeoJSON...")
        
        predictions = []
        confidences = []
        all_probs = []
        
        # Create progress tracking
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
                    # Make prediction
                    pred_class, confidence, probs = predict_single_image(
                        ort_session, img, transform, idx_to_class, current_model
                    )
                    
                    predictions.append(pred_class)
                    confidences.append(confidence)
                    all_probs.append(probs)
                    
                else:
                    predictions.append("unknown")
                    confidences.append(0.0)
                    all_probs.append(np.zeros(len(idx_to_class)))
                    
            except Exception as e:
                st.warning(f"Error processing polygon {idx+1}: {str(e)}")
                predictions.append("unknown")
                confidences.append(0.0)
                all_probs.append(np.zeros(len(idx_to_class)))
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Add predictions to GeoDataFrame
        result_gdf = gdf.copy()
        result_gdf['prediction'] = predictions
        result_gdf['confidence'] = confidences
        result_gdf['model_used'] = current_model
        
        # Add probability columns
        for class_idx, class_name in idx_to_class.items():
            if current_model == "agga-v4":
                # Extract text between parentheses
                if '(' in class_name and ')' in class_name:
                    clean_class_name = class_name.split('(')[1].split(')')[0]
                else:
                    clean_class_name = class_name
            else:
                clean_class_name = class_name.split(' - ')[0] if ' - ' in class_name else class_name
            result_gdf[f'prob_{clean_class_name}'] = [float(p[class_idx]) if len(p) > class_idx else 0.0 for p in all_probs]
        
        return result_gdf
        
    except Exception as e:
        st.error(f"Error processing GeoJSON file: {str(e)}")
        return None

# ========================
# Tab Functions
# ========================
def draw_polygons_tab():
    """Tab for drawing polygons interactively with location search"""
    st.subheader("🗺️ Interactive Map with Location Search")
    
    # Initialize geocoding service
    geocode_service = initialize_geocode_service()
    
    # Location search section
    if geocode_service:
        st.markdown("### 🔍 Search Location")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search for a location:",
                placeholder="Enter address, place name, or coordinates...",
                help="Search for any location worldwide (e.g., 'Jakarta', 'Times Square NY', 'Monas Jakarta')"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="secondary", use_container_width=True)
        
        # Handle search
        search_location = None
        if search_button and search_query.strip():
            with st.spinner(f"Searching for '{search_query}'..."):
                lat, lon, formatted_address = geocode_service.geocode_address(search_query)
                
                if lat and lon:
                    search_location = (lat, lon, formatted_address)
                    st.success(f"📍 Found: {formatted_address}")
                    
                    # Store search result in session state for map centering
                    st.session_state.search_location = search_location
                    st.session_state.map_center = (lat, lon)
                    st.session_state.map_zoom = 15  # Closer zoom for search results
                else:
                    st.error(f"❌ Could not find location: '{search_query}'")
    else:
        st.info("🔍 Location search unavailable - Google Maps API key not configured")
    
    # Get map parameters (use search results if available, otherwise use defaults)
    if 'map_center' in st.session_state and 'map_zoom' in st.session_state:
        center_lat, center_lon = st.session_state.map_center
        zoom_start = st.session_state.map_zoom
    else:
        # Default to Jakarta
        center_lat, center_lon, zoom_start = -7.25, 112.75, 12
    
    # Reset location button
    if 'search_location' in st.session_state:
        if st.button("🏠 Reset to Default Location", help="Return to default Jakarta view"):
            # Clear search results
            if 'search_location' in st.session_state:
                del st.session_state.search_location
            if 'map_center' in st.session_state:
                del st.session_state.map_center
            if 'map_zoom' in st.session_state:
                del st.session_state.map_zoom
            st.rerun()
    
    # Fixed settings (same as training data) 
    img_zoom, img_scale = 17, 2  # Fixed to match training
    
    # Create and display map (back to original simple version)
    m = create_map(center_lat, center_lon, zoom_start)
    
    # Add search marker if available
    if 'search_location' in st.session_state:
        lat, lon, formatted_address = st.session_state.search_location
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(f"<b>📍 Search Result</b><br>{formatted_address}", max_width=300),
            tooltip=f"📍 {formatted_address}",
            icon=folium.Icon(color='green', icon='search')
        ).add_to(m)
    
    map_data = st_folium(
        m, 
        width=None,  # Use full width
        height=600,  # Increased height
        returned_objects=["all_drawings"],
        key="map_widget"  # Back to original key
    )
    
    # Process drawn polygons (back to original working code)
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
            current_model_name = MODEL_CONFIGS[st.session_state.current_model]["display_name"]
            if st.button(f"🤖 Predict Classifications using {current_model_name}", type="primary", use_container_width=True):
                predict_polygons(gdf, img_zoom, img_scale)
        else:
            st.info("👆 Draw some polygons on the map to get started")
    else:
        st.info("👆 Use the search bar above to find a location, then draw polygons on the map")

    # Results section
    st.markdown("---")
    st.subheader("📊 Results")
    
    if 'predictions' in st.session_state and st.session_state.predictions is not None and len(st.session_state.predictions) > 0:
        display_prediction_results()
    else:
        st.info("Draw polygons and click 'Predict' to see results here")

def upload_geojson_tab():
    """Tab for uploading and processing GeoJSON files"""
    st.subheader("📁 Upload GeoJSON for Batch Processing")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a GeoJSON file",
        type=['geojson', 'json'],
        help="Upload a GeoJSON file containing polygons to classify"
    )
    
    if uploaded_file is not None:
        try:
            # Preview the uploaded file
            gdf_preview = gpd.read_file(uploaded_file)
            gdf_preview = gdf_preview.to_crs('EPSG:4326')
            
            st.success(f"✅ Loaded GeoJSON with {len(gdf_preview)} features")
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", len(gdf_preview))
            with col2:
                st.metric("CRS", str(gdf_preview.crs))
            with col3:
                bounds = gdf_preview.total_bounds
                st.metric("Bounds", f"{bounds[0]:.3f}, {bounds[1]:.3f}")
            
            # Show column info
            st.write("**📋 Available Columns:**")
            cols_info = []
            for col in gdf_preview.columns:
                if col != 'geometry':
                    dtype = str(gdf_preview[col].dtype)
                    unique_count = gdf_preview[col].nunique()
                    cols_info.append(f"- **{col}** ({dtype}) - {unique_count} unique values")
            
            for info in cols_info[:5]:  # Show first 5 columns
                st.write(info)
            
            if len(cols_info) > 5:
                st.write(f"... and {len(cols_info) - 5} more columns")
            
            # Process button
            current_model_name = MODEL_CONFIGS[st.session_state.current_model]["display_name"]
            if st.button(f"🤖 Process with {current_model_name}", type="primary", use_container_width=True):
                # Get model components
                model_components = (
                    st.session_state.ort_session,
                    st.session_state.transform,
                    st.session_state.idx_to_class,
                    st.session_state.current_model
                )
                
                # Process the GeoJSON
                result_gdf = process_geojson_batch(uploaded_file, model_components)
                
                if result_gdf is not None:
                    # Store results
                    st.session_state.batch_predictions = result_gdf
                    st.session_state.batch_processed = True
                    
                    # Show summary
                    st.subheader("📊 Processing Summary")
                    successful_preds = sum(1 for p in result_gdf['prediction'] if p != "unknown")
                    failed_preds = len(result_gdf) - successful_preds
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(result_gdf))
                    with col2:
                        st.metric("Successful", successful_preds)
                    with col3:
                        st.metric("Failed", failed_preds)
                    
                    # Prediction distribution
                    pred_counts = Counter(result_gdf['prediction'])
                    st.write("**🏷️ Prediction Distribution:**")
                    for pred, count in pred_counts.items():
                        percentage = (count / len(result_gdf)) * 100
                        st.write(f"- **{pred}**: {count} ({percentage:.1f}%)")
        
        except Exception as e:
            st.error(f"Error reading GeoJSON file: {str(e)}")
    
    # Display results if available
    if 'batch_predictions' in st.session_state and st.session_state.batch_predictions is not None:
        st.markdown("---")
        st.subheader("📊 Batch Processing Results")
        
        result_gdf = st.session_state.batch_predictions
        
        # Show results table
        st.write("**📋 Results Table:**")
        display_cols = ['prediction', 'confidence']
        prob_cols = [col for col in result_gdf.columns if col.startswith('prob_')]
        display_cols.extend(prob_cols[:5])  # Show first 5 probability columns
        st.dataframe(result_gdf[display_cols], use_container_width=True)
        
        # Create and display results map
        st.write("**🗺️ Results Map:**")
        results_map = create_results_map(result_gdf)
        if results_map:
            st_folium(results_map, width=None, height=600, key="results_map")
        
        # Download button
        st.write("**💾 Download Results:**")
        
        # Prepare download data
        download_gdf = result_gdf.copy()
        
        # Ensure all data is serializable
        for col in download_gdf.columns:
            if col != 'geometry':
                if download_gdf[col].dtype == 'object':
                    download_gdf[col] = download_gdf[col].astype(str)
                elif 'float' in str(download_gdf[col].dtype):
                    download_gdf[col] = download_gdf[col].round(6)
        
        # Convert to GeoJSON
        geojson_data = download_gdf.to_json()
        current_model = st.session_state.current_model
        
        st.download_button(
            label="🗺️ Download Results as GeoJSON",
            data=geojson_data,
            file_name=f"batch_predictions_{current_model}.geojson",
            mime="application/json",
            help=f"Download batch processing results as GeoJSON"
        )
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
    st.markdown("Choose between interactive polygon drawing or batch GeoJSON processing")

    # Sidebar
    with st.sidebar:
        st.header("🗺️ AGGA Models")
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose Classification Model:",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: MODEL_CONFIGS[x]["display_name"],
            index=0,  # Default to agga-v2
            help="Select which model to use for classification"
        )
        
        # Display selected model info
        model_config = MODEL_CONFIGS[selected_model]
        st.markdown(f"**Selected:** {model_config['display_name']}")
        st.markdown(f"**Description:** {model_config['description']}")
        
        # Show class details
        if selected_model == "agga-v4":
            with st.expander("📋 Class Details"):
                class_labels = model_config['class_labels']
                for class_idx, label in class_labels.items():
                    st.markdown(f"**{class_idx}**: {label}")
        elif selected_model == "agga-v2":
            with st.expander("📋 Class Details"):
                classes = model_config['default_classes']
                for i, cls in enumerate(classes):
                    st.markdown(f"**{i+1}**: {cls}")
        
        st.markdown("---")
        st.markdown("### 📋 Usage Options")
        st.markdown("""
        **🎯 Interactive Drawing:**
        - 🔍 Search locations worldwide
        - Draw polygons on the map
        - Real-time classification
        - View captured images
        
        **📁 Batch Processing:**
        - Upload GeoJSON file
        - Classify all polygons
        - Download results with map visualization
        """)
        st.markdown("---")
        st.markdown("### 🔍 Location Search")
        if 'geocode_service' in st.session_state and st.session_state.geocode_service:
            st.markdown("✅ Google Maps API connected")
        else:
            st.markdown("❌ Google Maps API not configured")
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.markdown("**Image Capture Settings:**")
        st.markdown("- Zoom Level: 17 (fixed)")
        st.markdown("- Scale: 2x (fixed)")  
        st.markdown("- Image Size: 640×640 (fixed)")
        st.markdown("*Settings match training data*")

    # Load model (check if model changed or not loaded yet)
    if ('model_loaded' not in st.session_state or 
        'current_model' not in st.session_state or 
        st.session_state.current_model != selected_model):
        
        with st.spinner(f"Loading {MODEL_CONFIGS[selected_model]['display_name']}..."):
            ort_session, config, class_to_idx, idx_to_class, model_key = download_and_load_model(selected_model)
            transform = get_prediction_transforms(config['image_size'])
            
            st.session_state.ort_session = ort_session
            st.session_state.config = config
            st.session_state.class_to_idx = class_to_idx
            st.session_state.idx_to_class = idx_to_class
            st.session_state.transform = transform
            st.session_state.model_loaded = True
            st.session_state.current_model = selected_model
            
            # Clear previous predictions when model changes
            if 'predictions' in st.session_state:
                del st.session_state.predictions
            if 'captured_images' in st.session_state:
                del st.session_state.captured_images
            if 'batch_predictions' in st.session_state:
                del st.session_state.batch_predictions

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Interactive Drawing", "📁 Batch Processing", "🗳️ Voting Prediction"])
    
    with tab1:
        draw_polygons_tab()
    
    with tab2:
        upload_geojson_tab()
        
    with tab3:
        voting_prediction_tab()


def predict_polygons(gdf, zoom, scale):
    """Predict classifications for drawn polygons using ONNX"""
    
    # Get model components from session state
    ort_session = st.session_state.ort_session
    transform = st.session_state.transform
    idx_to_class = st.session_state.idx_to_class
    current_model = st.session_state.current_model
    
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
                    ort_session, img, transform, idx_to_class, current_model
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
    result_gdf['model_used'] = current_model
    
    # Add probability columns
    for class_idx, class_name in idx_to_class.items():
        # For AGGA-v4, extract just the label part for column names (e.g., "green" from "1 (green)")
        if current_model == "agga-v4":
            # Extract text between parentheses
            if '(' in class_name and ')' in class_name:
                clean_class_name = class_name.split('(')[1].split(')')[0]
            else:
                clean_class_name = class_name
        else:
            clean_class_name = class_name.split(' - ')[0] if ' - ' in class_name else class_name
        result_gdf[f'prob_{clean_class_name}'] = [float(p[class_idx]) if len(p) > class_idx else 0.0 for p in all_probs]
    
    st.session_state.predictions = result_gdf
    st.session_state.captured_images = captured_images

def display_prediction_results():
    """Display prediction results"""
    if 'predictions' not in st.session_state:
        return
    
    result_gdf = st.session_state.predictions
    captured_images = st.session_state.get('captured_images', [])
    current_model = st.session_state.get('current_model', 'unknown')
    
    # Show which model was used
    model_name = MODEL_CONFIGS.get(current_model, {}).get('display_name', 'Unknown Model')
    st.info(f"🤖 Results from: **{model_name}**")
    
    # Results table
    st.write("**📊 Detailed Results:**")
    display_cols = ['prediction', 'confidence']
    if len(result_gdf.columns) > 10:  # If has probability columns
        prob_cols = [col for col in result_gdf.columns if col.startswith('prob_')]
        display_cols.extend(prob_cols[:5])  # Show first 5 probability columns
    
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
        file_name=f"polygon_predictions_{current_model}.geojson",
        mime="application/json",
        help=f"Download polygons with classification results from {model_name} as GeoJSON"
    )

if __name__ == "__main__":
    main()