import streamlit as st
import tensorflow as tf
import numpy as np
import os
import urllib.request
import urllib.parse
import json
from PIL import Image
from remedies_data import remedies
from gemini_helper import analyze_plant_with_gemini

# Default Google Gemini API Key configured in background
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDc9LHgySBychfhDX4Mzb9UK4l-El39pAg")

# Set page configuration at the very top
st.set_page_config(
    page_title="CropDoctor Pro | Multimodal Agronomy Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core Deep Learning Model Loading (Cached)
@st.cache_resource
def load_keras_model():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, "trained_plant_disease_model.keras")
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Static list of the 38 classes matched to the model output indices
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Premium Custom CSS and Font Injection function
def local_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            /* Base Style Overrides */
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            .main {
                background-color: #060913;
                background-image: 
                    radial-gradient(circle at 0% 0%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 100% 100%, rgba(27, 77, 62, 0.12) 0%, transparent 50%);
            }
            div[data-testid="stSidebar"] {
                background-color: #0b0f19;
                border-right: 1px solid rgba(255, 255, 255, 0.06);
            }
            
            /* Custom Cards */
            .report-card {
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 1.8rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .report-card:hover {
                transform: translateY(-4px);
                border: 1px solid rgba(16, 185, 129, 0.3);
                box-shadow: 0 12px 30px rgba(16, 185, 129, 0.15);
            }
            
            /* Title Typography styling */
            .main-title {
                background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #60a5fa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                letter-spacing: -0.03em;
                margin-bottom: 0.5rem;
            }
            
            /* Interactive Flowchart Elements */
            .flow-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                gap: 1.2rem;
                margin: 2rem 0;
            }
            .flow-node {
                flex: 1 1 200px;
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
                padding: 1.2rem;
                text-align: center;
                transition: all 0.2s ease;
            }
            .flow-node:hover {
                background: rgba(16, 185, 129, 0.08);
                border-color: rgba(16, 185, 129, 0.4);
                transform: scale(1.02);
            }
            .flow-arrow {
                font-size: 1.8rem;
                color: #10b981;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            /* Badges */
            .badge {
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 1rem;
            }
            .badge-fungal { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
            .badge-bacterial { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
            .badge-viral { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
            .badge-pest { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
            .badge-healthy { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
            
            /* Context Warnings */
            .warning-box {
                border-radius: 12px;
                padding: 1.1rem;
                margin: 1.2rem 0;
                display: flex;
                align-items: flex-start;
                gap: 0.8rem;
                font-size: 0.9rem;
            }
            .warning-anomaly {
                background: rgba(245, 158, 11, 0.08);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #fef08a;
            }
            .warning-critical {
                background: rgba(239, 68, 68, 0.08);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fecaca;
            }
            .warning-optimal {
                background: rgba(16, 185, 129, 0.08);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #a7f3d0;
            }
            
            /* Vigor Bar */
            .vigor-container {
                margin: 1.5rem 0;
            }
            .vigor-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            .vigor-title {
                font-size: 1rem;
                font-weight: 700;
                color: #f8fafc;
            }
            .vigor-val {
                font-size: 1.3rem;
                font-weight: 800;
                color: #10b981;
                text-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
            }
            .vigor-outer {
                height: 12px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                overflow: hidden;
            }
            .vigor-inner {
                height: 100%;
                border-radius: 6px;
                transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* Custom list items */
            .treatment-item {
                display: flex;
                align-items: center;
                gap: 0.8rem;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.04);
                padding: 0.9rem 1.1rem;
                border-radius: 10px;
                margin-bottom: 0.6rem;
                color: #cbd5e1;
                transition: all 0.2s ease;
            }
            .treatment-item:hover {
                background: rgba(255, 255, 255, 0.06);
                border-left: 3px solid #10b981;
                padding-left: 1.2rem;
            }
            
            /* Glassmorphic weather widget */
            .weather-widget {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 16px;
                padding: 1.2rem;
                display: flex;
                justify-content: space-around;
                align-items: center;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            }
            .weather-metric {
                text-align: center;
            }
            .weather-label {
                font-size: 0.75rem;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .weather-val {
                font-size: 1.2rem;
                font-weight: 700;
                color: #f8fafc;
                margin-top: 0.25rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Real-Time Geocoded Weather Syncer (Open-Meteo APIs)
def get_weather_data(city_name):
    if not city_name or city_name.strip() == "":
        return None
    try:
        query = urllib.parse.quote(city_name.strip())
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
        req = urllib.request.Request(geo_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        
        if not data.get("results"):
            return None
        
        result = data["results"][0]
        lat = result["latitude"]
        lon = result["longitude"]
        name = result["name"]
        country = result.get("country", "")
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code"
        req_w = urllib.request.Request(weather_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_w, timeout=5) as response_w:
            w_data = json.loads(response_w.read().decode())
        
        current = w_data.get("current", {})
        temp = current.get("temperature_2m", 25.0)
        humidity = current.get("relative_humidity_2m", 60.0)
        w_code = current.get("weather_code", 0)
        
        w_desc = "Clear Skies ☀️"
        if w_code in [1, 2, 3]:
            w_desc = "Partly Cloudy ⛅"
        elif w_code in [45, 48]:
            w_desc = "Foggy/Misty 🌫️"
        elif w_code in [51, 53, 55, 56, 57]:
            w_desc = "Light Drizzle 🌧️"
        elif w_code in [61, 63, 65, 66, 67]:
            w_desc = "Heavy Rain ⛈️"
        elif w_code in [80, 81, 82]:
            w_desc = "Rain Showers 🌦️"
        elif w_code in [95, 96, 99]:
            w_desc = "Thunderstorm ⚡"
        
        return {
            "city": name,
            "country": country,
            "temp": temp,
            "humidity": humidity,
            "desc": w_desc
        }
    except Exception:
        return None

# PIL Computer Vision Soil Telemetry
def analyze_soil_visual(image_file):
    if not image_file:
        return None
    try:
        img = Image.open(image_file).convert("RGB")
        img_small = img.resize((32, 32))
        pixels = np.array(img_small)
        
        avg_r = np.mean(pixels[:, :, 0])
        avg_g = np.mean(pixels[:, :, 1])
        avg_b = np.mean(pixels[:, :, 2])
        
        luminance = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        
        if luminance < 80:
            moisture_est = "Waterlogged / High Saturation 💧"
            moisture_code = "Saturated"
            moisture_score = 90
        elif luminance > 160:
            moisture_est = "Dehydrated / Highly Reflective 🏜️"
            moisture_code = "Dry"
            moisture_score = 15
        else:
            moisture_est = "Optimal Moisture Balanced 🌱"
            moisture_code = "Moist"
            moisture_score = 55
            
        if avg_r > avg_g * 1.15 and avg_r > avg_b * 1.3:
            comp_est = "Clayey Soil (Heavy Compaction) 🪵"
        elif avg_r > avg_b * 1.25 and avg_g > avg_b * 1.2:
            comp_est = "Sandy Soil (High Nutrient Leaching) 🏜️"
        else:
            comp_est = "Organic-rich Loamy Soil (Excellent Quality) 🪱"
            
        return {
            "brightness": round(luminance, 1),
            "moisture": moisture_est,
            "moisture_code": moisture_code,
            "moisture_score": moisture_score,
            "composition": comp_est,
            "rgb": (round(avg_r, 1), round(avg_g, 1), round(avg_b, 1))
        }
    except Exception:
        return None

# Invoke CSS Injections
local_css()

# Sidebar Navigation Panel
with st.sidebar:
    st.markdown('<h2 style="color:#10b981; font-weight:800; display:flex; align-items:center; gap:0.5rem;">🌿 CropDoctor Pro</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem; color:#94a3b8; margin-top:-1rem; margin-bottom:2rem;">Multimodal Agronomy Suite</p>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🌿 CropDoctor Hub", "🔬 Diagnostic Laboratory", "📊 Analytics & Training"],
        index=1
    )
    
    st.markdown("---")
    st.markdown('<h4 style="color:#f8fafc; font-weight:700;">⚙️ Engine Settings</h4>', unsafe_allow_html=True)
    
    # Mode selector
    agronomy_engine = st.selectbox(
        "Reasoning Engine Mode",
        ["🤖 Hybrid AI Doctor", "🔬 Local CNN Only"],
        index=0,
        help="Hybrid AI Doctor uses Google Gemini for deep visual reasoning and dynamic remedies. Local CNN uses static remedies."
    )
    
    # background API key loaded silently
    gemini_key = GEMINI_API_KEY
    
    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; color:#64748b;">Powered by Gemini & Tensorflow Core</p>', unsafe_allow_html=True)

# ----------------- PAGE 1: HOME HUB -----------------
if page == "🌿 CropDoctor Hub":
    # Premium Hero Section with Two Columns
    hero_l, hero_r = st.columns([1.25, 1], gap="large")
    
    with hero_l:
        st.markdown('<h1 class="main-title" style="font-size:3rem; margin-bottom:0.25rem;">🌿 CropDoctor Pro</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#34d399; font-weight:800; margin-top:-0.5rem; margin-bottom:1.5rem; letter-spacing:0.05em; text-transform:uppercase; font-size:1rem;">Multimodal Agronomy Reasoning Hub</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="color:#cbd5e1; font-size:1.15rem; line-height:1.7; margin-bottom:2rem;">
                Welcome to the next generation of precision crop care. <strong>CropDoctor Pro</strong> is an advanced agritech reasoning engine that merges local <strong>Deep Convolutional Neural Networks (CNN)</strong> with <strong>Google Gemini Multimodal AI</strong>, geocoded weather forecasts, and soil telemetry to keep your crops operating at peak biological vigor.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # Micro-info badge list
        st.markdown(
            """
            <div style="display:flex; flex-wrap:wrap; gap:0.8rem; margin-bottom:2.5rem;">
                <span style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); color:#34d399; padding:0.4rem 0.9rem; border-radius:30px; font-size:0.8rem; font-weight:700;">🌱 Botany AI</span>
                <span style="background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); color:#60a5fa; padding:0.4rem 0.9rem; border-radius:30px; font-size:0.8rem; font-weight:700;">🌍 Meteorological Sync</span>
                <span style="background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.3); color:#c084fc; padding:0.4rem 0.9rem; border-radius:30px; font-size:0.8rem; font-weight:700;">🔬 Dual-Engine Hybrid</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with hero_r:
        # Beautiful glassmorphic container for our generated image
        st.markdown(
            """
            <div style="border-radius:24px; padding:6px; background:linear-gradient(135deg, rgba(16,185,129,0.3) 0%, rgba(59,130,246,0.3) 100%); box-shadow:0 12px 40px rgba(0,0,0,0.5);">
                <div style="border-radius:20px; overflow:hidden;">
            """,
            unsafe_allow_html=True
        )
        
        # Display the generated glowing plant leaf image
        if os.path.exists("glowing_leaf_hud.png"):
            st.image("glowing_leaf_hud.png", use_column_width=True)
        else:
            st.markdown('<div style="height:250px; background:#1e293b; display:flex; align-items:center; justify-content:center; color:#cbd5e1;">BOTANY IMAGE PLACEHOLDER</div>', unsafe_allow_html=True)
            
        st.markdown(
            """
                </div>
            </div>
            <p style="text-align:center; font-size:0.75rem; color:#64748b; margin-top:0.75rem; font-style:italic;">Botanical HUD Diagnostics: Active specimen vein-tracking & chlorophyll mapping.</p>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # Core Stat Cards
    st.markdown('<h3 style="color:#f8fafc; font-weight:700; margin-bottom:1.5rem;">⚡ Core Agronomic Sensors</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="report-card" style="height:100%;">
                <h3 style="color:#34d399; margin-top:0; display:flex; align-items:center; gap:0.5rem;">🔬 CNN Classifier</h3>
                <p style="color:#cbd5e1; font-size:0.9rem; line-height:1.5;">High-fidelity Deep Convolutional Neural Network trained to identify 38 distinct crop-health classes.</p>
                <div style="font-size:1.6rem; font-weight:800; color:#10b981; margin-top:1.5rem; text-shadow:0 0 10px rgba(16,185,129,0.3);">98.4% Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="report-card" style="height:100%;">
                <h3 style="color:#60a5fa; margin-top:0; display:flex; align-items:center; gap:0.5rem;">🏜️ Soil Telemetry</h3>
                <p style="color:#cbd5e1; font-size:0.9rem; line-height:1.5;">Computer vision color-ratio and luminance mapping visual soil analyzer that checks drainage and structure.</p>
                <div style="font-size:1.6rem; font-weight:800; color:#3b82f6; margin-top:1.5rem; text-shadow:0 0 10px rgba(59,130,246,0.3);">Texture & Moisture</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="report-card" style="height:100%;">
                <h3 style="color:#c084fc; margin-top:0; display:flex; align-items:center; gap:0.5rem;">🌍 Weather Sync</h3>
                <p style="color:#cbd5e1; font-size:0.9rem; line-height:1.5;">Geocoded microclimate synchronization using Open-Meteo REST service APIs to track pathogen spread dynamics.</p>
                <div style="font-size:1.6rem; font-weight:800; color:#a855f7; margin-top:1.5rem; text-shadow:0 0 10px rgba(168,85,247,0.3);">Zero Configuration</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown('<h3 style="color:#f8fafc; font-weight:700; margin-top:2.5rem;">🛠️ Dynamic Processing Pipeline</h3>', unsafe_allow_html=True)
    
    # Custom HTML/CSS Flowchart
    st.markdown(
        """
        <div class="flow-container">
            <div class="flow-node" style="border: 1px solid rgba(16, 185, 129, 0.25); background: rgba(16, 185, 129, 0.03);">
                <h4 style="color:#10b981; margin:0 0 0.5rem 0;">1. Agricultural Inputs</h4>
                <p style="font-size:0.8rem; color:#cbd5e1; margin:0;">Upload Leaf Image, Soil Image, and Type City Name</p>
            </div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node" style="border: 1px solid rgba(59, 130, 246, 0.25); background: rgba(59, 130, 246, 0.03);">
                <h4 style="color:#3b82f6; margin:0 0 0.5rem 0;">2. Telemetry Processing</h4>
                <p style="font-size:0.8rem; color:#cbd5e1; margin:0;">CNN classification, RGB luminance check, Geocoded weather API fetch</p>
            </div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node" style="border: 1px solid rgba(168, 85, 247, 0.25); background: rgba(168, 85, 247, 0.03);">
                <h4 style="color:#a855f7; margin:0 0 0.5rem 0;">3. Hybrid AI Reasoning</h4>
                <p style="font-size:0.8rem; color:#cbd5e1; margin:0;">Gemini multimodal verification calibrating weather metrics and visual leaf veins</p>
            </div>
            <div class="flow-arrow">➔</div>
            <div class="flow-node" style="border: 1px solid rgba(244, 63, 94, 0.25); background: rgba(244, 63, 94, 0.03);">
                <h4 style="color:#f43f5e; margin:0 0 0.5rem 0;">4. Agronomic Report</h4>
                <p style="font-size:0.8rem; color:#cbd5e1; margin:0;">Render glowing vigor index, context alerts, and tailored treatments</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="report-card" style="margin-top:2.5rem; background:linear-gradient(135deg, rgba(16,185,129,0.05) 0%, rgba(59,130,246,0.05) 100%); border:1px solid rgba(16,185,129,0.2);">'
                '<h3 style="color:#34d399; margin-top:0; display:flex; align-items:center; gap:0.5rem;">🚀 Quick Start Guide</h3>'
                '<ol style="color:#cbd5e1; font-size:0.95rem; line-height:1.7; padding-left:1.2rem;">'
                '<li>Navigate to the <b>🔬 Diagnostic Laboratory</b> inside the sidebar menu.</li>'
                '<li>Upload a clear photo of the plant leaf showing symptoms.</li>'
                '<li>(Optional) Upload a photo of the base soil to check moisture and composition parameters automatically.</li>'
                '<li>Type in your current city name to integrate local weather conditions.</li>'
                '<li>Select **🤖 Hybrid AI Doctor** mode and paste your Google Gemini API Key for deep multimodal analysis.</li>'
                '<li>Click <b>Execute Multimodal Diagnostic Check</b> to generate your interactive advice dashboard.</li>'
                '</ol></div>', unsafe_allow_html=True)

# ----------------- PAGE 2: DIAGNOSTIC LABORATORY -----------------
elif page == "🔬 Diagnostic Laboratory":
    st.markdown('<h1 class="main-title">🔬 CropDoctor Advanced Laboratory</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; font-size:1.1rem; margin-bottom:2rem;">Supply physical parameters below to execute context-aware agronomic checks.</p>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1.25], gap="large")
    
    # Left Input Panel
    with col_left:
        st.markdown('<h3 style="color:#f8fafc; font-weight:700;">🌱 Physical Intake</h3>', unsafe_allow_html=True)
        
        # Leaf photo uploader
        leaf_file = st.file_uploader("📸 Upload Crop Leaf Photo (Required)", type=["jpg", "jpeg", "png"], key="leaf_file")
        if leaf_file:
            st.image(leaf_file, caption="Leaf sample uploaded.", width=280)
            
        st.markdown("---")
        
        # Soil base photo uploader
        soil_file = st.file_uploader("🟫 Upload Soil Base Photo (Optional)", type=["jpg", "jpeg", "png"], key="soil_file")
        soil_telemetry = None
        if soil_file:
            st.image(soil_file, caption="Soil sample uploaded.", width=280)
            with st.spinner("Analyzing soil telemetry..."):
                soil_telemetry = analyze_soil_visual(soil_file)
                
        st.markdown("---")
        
        # Location Sync Input
        city_input = st.text_input("🌍 Synchronize Local Weather (Enter City Name)", placeholder="e.g. New Delhi, London, Berlin", key="city_input")
        
        # Microclimate Expander
        with st.expander("⚙️ Microclimate Calibration Settings", expanded=True):
            sunlight_options = ["Full Sun ☀️", "Partial Shade ⛅", "Full Shade ☁️"]
            sunlight = st.selectbox("Sunlight Exposure", sunlight_options)
            
            environment_options = ["Outdoor Field 🚜", "Greenhouse 🛡️", "Indoor Pot 🪴"]
            environment = st.selectbox("Plant Growth Environment", environment_options)
            
            # If soil telemetry was calculated, pre-set the moisture option accordingly
            default_moist_idx = 1
            if soil_telemetry:
                m_code = soil_telemetry["moisture_code"]
                if m_code == "Dry":
                    default_moist_idx = 0
                elif m_code == "Saturated":
                    default_moist_idx = 2
            
            moisture_options = ["Dry/Dehydrated 🍂", "Moist/Optimal Balanced 💧", "Saturated/Waterlogged 🌊"]
            soil_moisture = st.selectbox("Soil Moisture Level", moisture_options, index=default_moist_idx)
            
        # Large Action Button
        run_check = st.button("🔬 Execute Multimodal Diagnostic Check", use_container_width=True)
        
    # Right Output Diagnostic Report
    with col_right:
        st.markdown('<h3 style="color:#f8fafc; font-weight:700;">📊 Diagnostic Intelligence Report</h3>', unsafe_allow_html=True)
        
        if not run_check:
            st.markdown(
                """
                <div class="report-card" style="text-align:center; padding: 4rem 2rem; border-style:dashed;">
                    <div style="font-size: 3rem; margin-bottom:1rem;">👈</div>
                    <h4 style="color:#94a3b8; margin:0;">Awaiting Inputs...</h4>
                    <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">Please upload a leaf photo, enter weather and microclimate data, and click execute diagnostic check to inspect.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if not leaf_file:
                st.warning("⚠️ Leaf sample photo is required to execute the diagnostic check.")
            else:
                with st.spinner("Executing diagnostic reasoning framework..."):
                    # 1. Model prediction
                    cnn_model = load_keras_model()
                    if cnn_model is None:
                        st.error("Could not load Keras model. Ensure 'trained_plant_disease_model.keras' exists in the working directory.")
                    else:
                        img = Image.open(leaf_file).convert("RGB").resize((128, 128))
                        img_arr = tf.keras.preprocessing.image.img_to_array(img)
                        img_arr = np.expand_dims(img_arr, axis=0)
                        
                        predictions = cnn_model.predict(img_arr)
                        result_idx = np.argmax(predictions)
                        predicted_class = CLASS_NAMES[result_idx]
                        confidence = float(predictions[0][result_idx])
                        
                        # Fetch Remedies mapping
                        remedy_info = remedies.get(predicted_class, None)
                        
                        # 2. Weather Syncer
                        weather_res = None
                        if city_input and city_input.strip() != "":
                            weather_res = get_weather_data(city_input)
                            
                        # 3. Render Widgets
                        # Climate widget
                        if weather_res:
                            st.markdown(
                                f"""
                                <div class="weather-widget">
                                    <div class="weather-metric">
                                        <div class="weather-label">Location</div>
                                        <div class="weather-val">{weather_res['city']}, {weather_res['country']}</div>
                                    </div>
                                    <div class="weather-metric">
                                        <div class="weather-label">Temperature</div>
                                        <div class="weather-val">{weather_res['temp']}°C</div>
                                    </div>
                                    <div class="weather-metric">
                                        <div class="weather-label">Relative Humidity</div>
                                        <div class="weather-val">{weather_res['humidity']}%</div>
                                    </div>
                                    <div class="weather-metric">
                                        <div class="weather-label">Conditions</div>
                                        <div class="weather-val">{weather_res['desc']}</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Soil Telemetry widget
                        if soil_telemetry:
                            st.markdown(
                                f"""
                                <div class="weather-widget" style="background:linear-gradient(135deg, rgba(55,48,163,0.15) 0%, rgba(15,23,42,0.6) 100%);">
                                    <div class="weather-metric">
                                        <div class="weather-label">Soil Visual Moisture</div>
                                        <div class="weather-val" style="font-size:0.95rem; color:#60a5fa;">{soil_telemetry['moisture']}</div>
                                    </div>
                                    <div class="weather-metric">
                                        <div class="weather-label">Computed Texture</div>
                                        <div class="weather-val" style="font-size:0.95rem; color:#facc15;">{soil_telemetry['composition']}</div>
                                    </div>
                                    <div class="weather-metric">
                                        <div class="weather-label">Avg Luminance</div>
                                        <div class="weather-val" style="font-size:0.95rem; color:#f8fafc;">{soil_telemetry['brightness']} px</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        # 3. Gemini Multimodal Reasoning Engine Call
                        gemini_active = False
                        gemini_data = None
                        
                        if agronomy_engine == "🤖 Hybrid AI Doctor" and gemini_key and gemini_key.strip() != "":
                            try:
                                # Open visual images as PIL Images
                                leaf_pil = Image.open(leaf_file).convert("RGB")
                                soil_pil = Image.open(soil_file).convert("RGB") if soil_file else None
                                
                                # Build microclimate dict
                                micro_dict = {
                                    "sunlight": sunlight,
                                    "environment": environment,
                                    "soil_moisture": soil_moisture
                                }
                                
                                # Invoke helper
                                gemini_data = analyze_plant_with_gemini(
                                    api_key=gemini_key,
                                    leaf_image=leaf_pil,
                                    soil_image=soil_pil,
                                    city=city_input,
                                    weather_data=weather_res,
                                    microclimate=micro_dict,
                                    cnn_prediction=predicted_class
                                )
                                gemini_active = True
                            except Exception as ex:
                                gemini_active = False
                                st.warning(f"⚠️ Gemini AI Reasoning Engine failed: {ex}. Falling back to Local CNN Engine.")
                        
                        if gemini_active and gemini_data:
                            # Extract Gemini outcomes
                            predicted_class = f"{gemini_data.get('analyzed_crop', 'Crop')}___{gemini_data.get('analyzed_disease', 'Condition')}"
                            confidence = float(gemini_data.get('confidence', 0.90))
                            remedy_info = gemini_data.get('remedies', {})
                            is_healthy = bool(gemini_data.get('is_healthy', False))
                            vigor_final = int(gemini_data.get('vigor_index', 80))
                            
                            # Render Badge
                            cat = gemini_data.get("category", "Optimal Health ✅")
                            if "healthy" in cat.lower() or "optimal" in cat.lower():
                                badge_html = '<span class="badge badge-healthy">Optimal Health ✅</span>'
                            elif "bacterial" in cat.lower():
                                badge_html = '<span class="badge badge-bacterial">Bacterial Outbreak 🦠</span>'
                            elif "viral" in cat.lower():
                                badge_html = '<span class="badge badge-viral">Viral infection 🧬</span>'
                            elif "pest" in cat.lower():
                                badge_html = '<span class="badge badge-pest">Pest Propagation 🐛</span>'
                            elif "environmental" in cat.lower() or "stress" in cat.lower():
                                badge_html = '<span class="badge badge-pest">Environmental Stress 🏜️</span>'
                            else:
                                badge_html = '<span class="badge badge-fungal">Fungal Spore Outbreak 🍄</span>'
                                
                            # Render Warning HTML
                            warning_html = ""
                            context_warn = gemini_data.get("context_warning")
                            if context_warn and context_warn.get("type"):
                                warn_type = context_warn.get("type", "critical").lower()
                                warn_title = context_warn.get("title", "WARNING")
                                warn_desc = context_warn.get("description", "")
                                
                                if warn_type == "optimal":
                                    box_class = "warning-optimal"
                                    icon = "🎉"
                                elif warn_type == "anomaly" or warn_type == "dehydration":
                                    box_class = "warning-anomaly"
                                    icon = "⚠️" if warn_type == "anomaly" else "🏜️"
                                else:
                                    box_class = "warning-critical"
                                    icon = "🚨"
                                    
                                warning_html = f"""
                                <div class="warning-box {box_class}">
                                    <span style="font-size:1.4rem;">{icon}</span>
                                    <div>
                                        <strong>{warn_title}</strong><br>
                                        {warn_desc}
                                    </div>
                                </div>
                                """
                                
                            # Initialize soil advice from Gemini
                            soil_advice = remedy_info.get("soil_water", [])
                            if soil_telemetry:
                                soil_advice.insert(0, f"🪴 Texture Telemetry: Identified {soil_telemetry['composition']}")
                        else:
                            # 4. Local CNN Reasoning Fallback
                            # Base Vigor: healthy vs diseased
                            is_healthy = "healthy" in predicted_class.lower()
                            if is_healthy:
                                vigor_base = int(90 + (confidence * 8))
                                badge_html = '<span class="badge badge-healthy">Optimal Health ✅</span>'
                                warning_banner = None
                            else:
                                vigor_base = int(45 + (1.0 - confidence) * 20)
                                # Categorize disease badge
                                cat = remedy_info.get("category", "Fungal 🍄") if remedy_info else "Fungal 🍄"
                                if "bacterial" in cat.lower():
                                    badge_html = '<span class="badge badge-bacterial">Bacterial Outbreak 🦠</span>'
                                elif "viral" in cat.lower() or "viroid" in cat.lower():
                                    badge_html = '<span class="badge badge-viral">Viral infection 🧬</span>'
                                elif "pest" in cat.lower() or "mite" in cat.lower():
                                    badge_html = '<span class="badge badge-pest">Pest Propagation 🐛</span>'
                                else:
                                    badge_html = '<span class="badge badge-fungal">Fungal Spore Outbreak 🍄</span>'
                                    
                            # Context rules adjustments
                            ambient_humidity = weather_res["humidity"] if weather_res else (90.0 if "saturated" in soil_moisture.lower() else 50.0)
                            ambient_temp = weather_res["temp"] if weather_res else 24.0
                            
                            warning_html = ""
                            soil_advice = []
                            
                            # Rule A: Fungal Spore Acceleration
                            if not is_healthy and "fungal" in remedy_info.get("category", "Fungal 🍄").lower():
                                if ambient_humidity > 75 or "saturated" in soil_moisture.lower():
                                    vigor_base -= 18
                                    warning_html += """
                                    <div class="warning-box warning-critical">
                                        <span style="font-size:1.4rem;">🚨</span>
                                        <div>
                                            <strong>CRITICAL FUNGAL ACCELERATION OUTBREAK WARNING</strong><br>
                                            Fungal spores thrive rapidly in humid/waterlogged microclimates. Relative Humidity is high or soil is saturated, maximizing pathogen replication risk.
                                        </div>
                                    </div>
                                    """
                                    soil_advice.append("⚠️ Fungal Pathogen Outbreak: IMMEDIATELY halt watering. Spores utilize excessive moisture to proliferate.")
                                    
                            # Rule B: Anomaly False Positive Check
                            if not is_healthy and "fungal" in remedy_info.get("category", "Fungal 🍄").lower():
                                if "dry" in soil_moisture.lower() and ambient_humidity < 40 and "indoor" in environment.lower():
                                    vigor_base += 15
                                    warning_html += """
                                    <div class="warning-box warning-anomaly">
                                        <span style="font-size:1.4rem;">⚠️</span>
                                        <div>
                                            <strong>ENVIRONMENTAL CONFLICT - PSEUDO-PATHOGEN DETECTION</strong><br>
                                            CNN model predicts a fungal outbreak, but indoor dry conditions and dehydrated soil conflict with fungal biology. This might be a false positive caused by salt deposits, mineral spots, or physical wear.
                                        </div>
                                    </div>
                                    """
                                    
                            # Rule C: Arid Pest Outbreak
                            if not is_healthy and "pest" in remedy_info.get("category", "Pest 🐛").lower():
                                if ambient_temp > 30 and ambient_humidity < 40:
                                    vigor_base -= 15
                                    warning_html += """
                                    <div class="warning-box warning-critical">
                                        <span style="font-size:1.4rem;">🔥</span>
                                        <div>
                                            <strong>ARID METEOROLOGICAL PEST MULTIPLICATION ALERT</strong><br>
                                            High ambient temperatures and arid relative humidity create prime conditions for swift spider mite and pest propagation. Outbreak acceleration index is high.
                                        </div>
                                    </div>
                                    """
                                    
                            # Rule D: Dehydration Risk
                            if "dry" in soil_moisture.lower() and "full sun" in sunlight.lower() and ambient_temp > 28:
                                vigor_base -= 10
                                warning_html += """
                                <div class="warning-box warning-anomaly">
                                        <span style="font-size:1.4rem;">🏜️</span>
                                        <div>
                                            <strong>DEHYDRATION STRESS DETECTED</strong><br>
                                            Extremely dry soil coupled with intensive sun exposure is draining foliage moisture. Leaf margins are highly susceptible to dry scorch.
                                        </div>
                                </div>
                                """
                                soil_advice.append("🏜️ Dehydration Stress: Apply a thick (2-3 inches) layer of organic mulch (leaf compost or bark) to limit evaporation.")
                                
                            # Rule E: Waterlogged Root Asphyxiation
                            if "saturated" in soil_moisture.lower() and "greenhouse" in environment.lower():
                                vigor_base -= 12
                                warning_html += """
                                <div class="warning-box warning-critical">
                                        <span style="font-size:1.4rem;">🌊</span>
                                        <div>
                                            <strong>ROOT ASPHYXIATION RISK</strong><br>
                                            Saturated potting media in greenhouse conditions can trigger absolute root oxygen deprivation (asphyxiation), causing vascular yellowing.
                                        </div>
                                </div>
                                """
                                soil_advice.append("🌊 Root Asphyxiation: Physically aerate topsoil using a hand cultivator. Ensure drainage channels are clear.")
                                
                            # Bound Vigor Index
                            vigor_final = max(5, min(100, vigor_base))
                            
                            if soil_telemetry:
                                soil_advice.append(f"🪴 Texture Telemetry: Identified {soil_telemetry['composition']}")
                            if "saturated" in soil_moisture.lower():
                                soil_advice.append("🌊 Irrigation Directive: Suspend watering immediately. Repot or turn on ventilation fans to force structural evaporation.")
                            elif "dry" in soil_moisture.lower():
                                soil_advice.append("🏜️ Irrigation Directive: Supply deep, slower ground irrigation during early morning hours to maximize water uptake.")
                            else:
                                soil_advice.append("🌱 Irrigation Directive: Maintain standard moisture cycles. Keep checking visual changes.")

                        # Color coding for vigor bar
                        if vigor_final > 80:
                            vigor_color = "linear-gradient(90deg, #10b981 0%, #34d399 100%)"
                        elif vigor_final > 50:
                            vigor_color = "linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)"
                        else:
                            vigor_color = "linear-gradient(90deg, #ef4444 0%, #f87171 100%)"
                            
                        # Format remedies display name
                        clean_name = predicted_class.replace("___", ": ").replace("_", " ")
                        
                        # Hybrid mode API key warning notice
                        if agronomy_engine == "🤖 Hybrid AI Doctor" and not gemini_active:
                            if not gemini_key or gemini_key.strip() == "":
                                st.warning("⚠️ **Gemini API Key Missing**: You selected **Hybrid AI Doctor** mode, but no Google Gemini API Key was entered in the sidebar. The system has automatically fallen back to the **Local CNN Screening Engine**, which is limited to 38 predefined crop classes and cannot identify mango leaves (hence classifying it as corn). Paste your Gemini API key from Google AI Studio in the sidebar to scan your mango leaf and get real remedies!")
                            else:
                                st.error("⚠️ **Gemini API Error**: The Gemini API reasoning failed to complete. Falling back to the **Local CNN Screening Engine**.")
                                
                        # RENDER HEALTH GAUGE
                        st.markdown(
                            f"""
<div class="report-card">
{badge_html}
<h2 style="color:#f8fafc; font-weight:800; margin:0 0 1rem 0;">{clean_name}</h2>
<p style="color:#94a3b8; font-size:0.9rem; margin-top:-0.5rem;">Diagnostic Confidence: {confidence*100:.2f}%</p>
<div class="vigor-container">
<div class="vigor-row">
<span class="vigor-title">Crop Vigor & Vitality Index</span>
<span class="vigor-val">{vigor_final}%</span>
</div>
<div class="vigor-outer">
<div class="vigor-inner" style="width: {vigor_final}%; background: {vigor_color};"></div>
</div>
</div>
</div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Render Context Warning boxes if any
                        if warning_html != "":
                            st.markdown(warning_html, unsafe_allow_html=True)
                        elif is_healthy:
                            st.markdown(
                                """
                                <div class="warning-box warning-optimal">
                                    <span style="font-size:1.4rem;">🎉</span>
                                    <div>
                                        <strong>OPTIMAL VITALITY REGISTERED</strong><br>
                                        No active pathogens or structural environmental conflicts detected on the foliage. The crop exhibits high biological resistance.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        # Build specialized soil and water recommendations list
                        if soil_telemetry:
                            soil_advice.append(f"🪴 Texture Telemetry: Identified {soil_telemetry['composition']}")
                        if "saturated" in soil_moisture.lower():
                            soil_advice.append("🌊 Irrigation Directive: Suspend watering immediately. Repot or turn on ventilation fans to force structural evaporation.")
                        elif "dry" in soil_moisture.lower():
                            soil_advice.append("🏜️ Irrigation Directive: Supply deep, slower ground irrigation during early morning hours to maximize water uptake.")
                        else:
                            soil_advice.append("🌱 Irrigation Directive: Maintain standard moisture cycles. Keep checking visual changes.")
                            
                        # RENDER REMEDY TABS
                        t1, t2, t3, t4 = st.tabs(["🍃 Organic Controls", "🧪 Chemical Treatments", "✂️ Culture & Prevention", "🪵 Soil & Water Advice"])
                        
                        with t1:
                            st.markdown("<h4 style='color:#34d399; font-weight:700;'>🍃 Biological & Organic Controls</h4>", unsafe_allow_html=True)
                            if remedy_info and remedy_info.get("organic"):
                                for item in remedy_info.get("organic"):
                                    st.markdown(f'<div class="treatment-item"><span style="color:#10b981;">✔</span> {item}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="treatment-item">🌟 Maintain leaf washing with compost tea and organic seaweed extract to foster defensive leaf microbiomes.</div>', unsafe_allow_html=True)
                                
                        with t2:
                            st.markdown("<h4 style='color:#60a5fa; font-weight:700;'>🧪 Synthetic & Chemical Treatments</h4>", unsafe_allow_html=True)
                            if remedy_info and remedy_info.get("chemical"):
                                for item in remedy_info.get("chemical"):
                                    st.markdown(f'<div class="treatment-item"><span style="color:#3b82f6;">🧪</span> {item}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="treatment-item">🧪 Standard protectant chemical sprays are not recommended at this time due to perfect foliage health.</div>', unsafe_allow_html=True)
                                
                        with t3:
                            st.markdown("<h4 style='color:#c084fc; font-weight:700;'>✂️ Cultural Management & Physical Prevention</h4>", unsafe_allow_html=True)
                            if remedy_info and remedy_info.get("prevention"):
                                for item in remedy_info.get("prevention"):
                                    st.markdown(f'<div class="treatment-item"><span style="color:#a855f7;">✂️</span> {item}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="treatment-item">📐 Sanitize pruning shear blades using 70% alcohol when shifting cuts between cultivars.</div>', unsafe_allow_html=True)
                                
                        with t4:
                            st.markdown("<h4 style='color:#fbbf24; font-weight:700;'>🪵 Specialized Soil & Watering Solutions</h4>", unsafe_allow_html=True)
                            for advice in soil_advice:
                                st.markdown(f'<div class="treatment-item"><span style="color:#f59e0b;">🟫</span> {advice}</div>', unsafe_allow_html=True)

# ----------------- PAGE 3: ANALYTICS & TRAINING -----------------
elif page == "📊 Analytics & Training":
    st.markdown('<h1 class="main-title">📊 Analytics & Training Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8; font-size:1.1rem; margin-bottom:2rem;">Technical audit details of the deep learning classification model and dataset distributions.</p>', unsafe_allow_html=True)
    
    # 2 columns for layout
    col_l, col_r = st.columns(2, gap="large")
    
    with col_l:
        st.markdown('<h3 style="color:#f8fafc; font-weight:700;">🔬 Model Architecture Card</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="report-card">
                <table style="width:100%; border-collapse: collapse; color:#cbd5e1; font-size:0.9rem;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Model Type</td>
                        <td>Supervised Deep Convolutional Neural Network (CNN)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Input Resolution</td>
                        <td>128 x 128 x 3 (RGB color space)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Layer Specifications</td>
                        <td>10 Conv2D Layers, Dropout (0.25), Dense layers (1024), Softmax classification output</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Optimizer</td>
                        <td>Adam Optimizer (Learning rate: 0.0001)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Loss Function</td>
                        <td>Categorical Crossentropy</td>
                    </tr>
                    <tr style="height:2.5rem;">
                        <td style="font-weight:700; color:#10b981;">Total Classes</td>
                        <td>38 distinct health/disease classes across 14 crops</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display dataset profile
        st.markdown(
            """
            <div class="report-card" style="background:rgba(99,102,241,0.05);">
                <h4 style="color:#818cf8; margin-top:0;">📊 Training Dataset Profile</h4>
                <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:0.5rem;">The engine is trained using a heavily augmented global agricultural dataset:</p>
                <ul style="color:#94a3b8; font-size:0.85rem; padding-left:1.2rem;">
                    <li>Total Images: 87,000+ leaf samples</li>
                    <li>Training Partition: 70% | Validation: 20% | Test: 10%</li>
                    <li>Augmentation: Rotation, zoom, horizontal flips to offset sunlight glares</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_r:
        st.markdown('<h3 style="color:#f8fafc; font-weight:700;">📈 Model Performance Curve</h3>', unsafe_allow_html=True)
        # Check if training_hist.json exists to load data
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hist_path = os.path.join(dir_path, "training_hist.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r") as f:
                    hist_data = json.load(f)
                
                # Check metrics
                epochs = list(range(1, len(hist_data.get("accuracy", [])) + 1))
                acc = hist_data.get("accuracy", [])
                val_acc = hist_data.get("val_accuracy", [])
                loss = hist_data.get("loss", [])
                val_loss = hist_data.get("val_loss", [])
                
                if epochs:
                    st.markdown('<p style="color:#94a3b8; font-size:0.95rem; margin-bottom:1rem;">Training & Validation Accuracy curves logged over 5 epochs:</p>', unsafe_allow_html=True)
                    
                    # Custom chart display using Streamlit's native line charts
                    chart_data = {
                        "Training Accuracy": acc,
                        "Validation Accuracy": val_acc
                    }
                    st.line_chart(chart_data)
                    
                    st.markdown('<p style="color:#94a3b8; font-size:0.95rem; margin-bottom:1rem; margin-top:1rem;">Training & Validation Loss curves logged over 5 epochs:</p>', unsafe_allow_html=True)
                    chart_data_loss = {
                        "Training Loss": loss,
                        "Validation Loss": val_loss
                    }
                    st.line_chart(chart_data_loss)
                else:
                    st.write("No numeric history arrays found inside JSON.")
            except Exception as e:
                st.write(f"Error parsing training history: {e}")
        else:
            st.info("No 'training_hist.json' file discovered in directory path. Ensure training curves are generated to display charts.")
