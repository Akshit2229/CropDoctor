# pyrefly: ignore [missing-import]
import google.generativeai as genai
import json
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_helper")

def analyze_plant_with_gemini(
    api_key: str,
    leaf_image: Image.Image,
    soil_image: Image.Image = None,
    city: str = None,
    weather_data: dict = None,
    microclimate: dict = None,
    cnn_prediction: str = None
) -> dict:
    """
    Multimodal agronomic reasoning engine using Google's Gemini Vision API.
    Sends leaf (and soil) photo along with local weather and microclimate data
    to receive high-fidelity, structured JSON remedies.
    """
    if not api_key or api_key.strip() == "":
        raise ValueError("Google Gemini API Key is missing or invalid.")

    # Configure Gemini API
    genai.configure(api_key=api_key.strip())
    
    # Use gemini-2.5-flash as default, fallback to gemini-1.5-flash if needed
    model_name = "gemini-2.5-flash"
    
    # Compile weather info
    weather_str = "None available"
    if weather_data:
        weather_str = f"Temp: {weather_data.get('temp', 25.0)}°C, Humidity: {weather_data.get('humidity', 60.0)}%, Conditions: {weather_data.get('desc', 'Clear')}"
        
    microclimate_str = "None available"
    if microclimate:
        microclimate_str = f"Sunlight: {microclimate.get('sunlight', 'N/A')}, Environment: {microclimate.get('environment', 'N/A')}, Soil Moisture: {microclimate.get('soil_moisture', 'N/A')}"

    # Build prompt
    prompt = f"""
You are an expert agronomic doctor, plant pathologist, and senior crop advisor.
Analyze the provided leaf image (and the base soil image if uploaded) under these exact conditions:
- Target City / Location: {city or 'Unknown'}
- Ambient Weather Telemetry: {weather_str}
- Microclimate Calibration: {microclimate_str}
- Screen CNN Model Prediction: {cnn_prediction or 'None'}

Please execute a rigorous multimodal diagnostic check:
1. Identify the crop type and assess the foliage health.
2. Cross-reference the screen CNN model prediction with your deep generative computer vision reasoning. If they match, confirm the finding. If they conflict, determine whether the CNN prediction is a false positive caused by environmental stress anomalies (e.g. fertilizer salts, mineral/nutrient deficiency, dry heat margins, pot bound limits) or a real biological outbreak (fungal, bacterial, viral, pest).
3. Compute a final 'vigor_index' (0-100) reflecting the crop's dynamic vitality. Healthy plants should score 85-100. Minor stress should score 60-80. Active pathogen outbreaks or severe structural dehydration/oxygen deprivation should score 10-50.
4. Construct a relevant 'context_warning' if there is an environmental clash, rapid pathogen propagation risk, or optimal state.
5. Generate real, high-quality, practical treatment remedies categorized into:
   - Biological & Organic Controls (natural, eco-friendly)
   - Synthetic & Chemical Treatments (precise fungicides, insecticides, active ingredients)
   - Cultural Management & Physical Prevention (crop rotation, pruning, spacing, watering method shifts)
   - Specialized Soil & Water Solutions (repotting, mulching, drainage aerating, moisture levels calibration)

You MUST respond with a strictly formatted JSON object. 
IMPORTANT: Do NOT wrap the JSON response inside markdown code block syntax (like ```json ... ```). Output the raw JSON text directly.
Ensure the response is completely valid JSON.

JSON Schema to follow:
{{
  "analyzed_crop": "string",
  "analyzed_disease": "string",
  "is_healthy": boolean,
  "confidence": float,
  "category": "string (Use EXACTLY one of: 'Fungal Outbreak 🍄', 'Bacterial Outbreak 🦠', 'Viral infection 🧬', 'Pest Propagation 🐛', 'Optimal Health ✅', 'Environmental Stress 🏜️')",
  "vigor_index": integer,
  "context_warning": {{
    "type": "string (one of: 'critical', 'anomaly', 'optimal', 'dehydration')",
    "title": "string (ALL CAPS short alert heading, e.g. CRITICAL FUNGAL OUTBREAK ACCELERATION)",
    "description": "string (1-2 sentences explaining the contextual risk based on humidity, temperature, soil moisture, and environment)"
  }},
  "remedies": {{
    "organic": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "chemical": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "prevention": [
      "string (remedy 1)",
      "string (remedy 2)"
    ],
    "soil_water": [
      "string (remedy 1)",
      "string (remedy 2)"
    ]
  }}
}}
"""

    contents = []
    
    # 1. Add prompt text
    contents.append(prompt)
    
    # 2. Add Leaf Image (convert PIL Image if it's not already)
    if isinstance(leaf_image, Image.Image):
        contents.append(leaf_image)
    else:
        logger.warning("leaf_image is not a valid PIL Image")
        
    # 3. Add Soil Image if provided
    if soil_image and isinstance(soil_image, Image.Image):
        contents.append(soil_image)
        logger.info("Soil image provided in multimodal payload")

    try:
        logger.info(f"Initializing Gemini GenerativeModel with {model_name}...")
        model = genai.GenerativeModel(model_name)
        
        logger.info("Executing generate_content...")
        response = model.generate_content(
            contents=contents,
            generation_config={"response_mime_type": "application/json"}
        )
        
        text_response = response.text.strip()
        
        # Clean potential markdown wrappers just in case the model ignored instructions
        if text_response.startswith("```"):
            lines = text_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            text_response = "\n".join(lines).strip()
            
        data = json.loads(text_response)
        logger.info("Successfully analyzed plant with Gemini API and parsed structured JSON.")
        return data
        
    except Exception as e:
        logger.error(f"Gemini API invocation failed: {e}")
        # Re-raise with user friendly message or handle in caller
        raise RuntimeError(f"Gemini AI reasoning failed: {str(e)}")
