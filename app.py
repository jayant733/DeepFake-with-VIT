import os
os.environ["HF_HOME"] = "/tmp"

from flask import Flask, request, jsonify, render_template
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import logging
from functools import wraps
from time import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
INFERENCE_TIMEOUT = 30  # seconds

# Model and processor (lazy loaded)
_model = None
_processor = None
_device = None

# Rate limiting
rate_limit_data = defaultdict(lambda: {'count': 0, 'reset_time': time()})
MAX_REQUESTS_PER_MINUTE = 60

def get_device():
    """Get appropriate device (GPU or CPU)"""
    global _device
    if _device is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {_device}")
    return _device

def load_model():
    """Lazy load model and processor"""
    global _model, _processor
    if _model is None:
        logger.info("Loading model...")
        try:
            device = get_device()
            _model = ViTForImageClassification.from_pretrained("Sxhni/deepfake-detector-vit")
            _model.to(device)
            _model.eval()  # Set to evaluation mode
            _processor = ViTImageProcessor.from_pretrained("Sxhni/deepfake-detector-vit")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return _model, _processor

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time()
        
        # Reset counter if minute has passed
        if current_time - rate_limit_data[client_ip]['reset_time'] > 60:
            rate_limit_data[client_ip] = {'count': 0, 'reset_time': current_time}
        
        # Check rate limit
        if rate_limit_data[client_ip]['count'] >= MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return jsonify({"error": "Rate limit exceeded. Maximum 60 requests per minute."}), 429
        
        rate_limit_data[client_ip]['count'] += 1
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model, processor = load_model()

@app.route("/predict", methods=["POST"])
@rate_limit
def predict():
    try:
        # Validate file presence
        if "image" not in request.files:
            logger.warning("No image file in request")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        
        # Validate filename
        if file.filename == "":
            logger.warning("Empty filename")
            return jsonify({"error": "No image selected"}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file extension: {file.filename}")
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024:.1f}MB"}), 413
        file.seek(0)  # Reset to beginning
        
        # Load model
        model, processor = load_model()
        device = get_device()
        
        # Open and validate image
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return jsonify({"error": "Invalid image format"}), 400
        
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess & forward pass
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

        # Find predicted label + confidence
        pred_id = probs.argmax()
        pred_label = model.config.id2label[pred_id]
        confidence = float(probs[pred_id])
        
        logger.info(f"Prediction: {pred_label} (confidence: {confidence:.2f})")

        return jsonify({
            "prediction": pred_label,
            "confidence": confidence,
            "all_probabilities": {
                model.config.id2label[i]: float(probs[i]) for i in range(len(probs))
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during processing. Please try again."}), 500

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    logger.info("Starting Deepfake Detector API")
    app.run(host="0.0.0.0", port=7860, debug=False)
