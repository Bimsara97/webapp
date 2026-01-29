#Import Libraries
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/history', exist_ok=True)

# Model configuration
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Disease class names (update based on your dataset)
CLASS_NAMES = ['Healthy', 'Downy Mildew', 'Leaf Curl', 'Rust/Mosaic']

# Load models
MODEL_PATH = 'models/mobilenetv3_small_converted.h5'
model = None

def load_model():
    """Load the trained model with compatibility handling"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            # Try loading with compile=False to avoid compatibility issues
            try:
                print("🔄 Loading model...")
                model = keras.models.load_model(MODEL_PATH, compile=False)
                
                # Recompile the model manually
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print("✅ Model loaded successfully!")
                return True
            except Exception as e1:
                print(f"⚠️  Standard loading failed: {str(e1)}")
                print("🔄 Trying alternative loading method...")
                
                # Try loading with custom objects
                import tensorflow as tf
                with tf.keras.utils.custom_object_scope({}):
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                print("✅ Model loaded with alternative method!")
                return True
        else:
            print(f"❌ Model file not found at {MODEL_PATH}")
            print("Please place your trained model (.h5 file) in the 'models/' directory")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("\n💡 Possible solutions:")
        print("   1. Ensure TensorFlow version matches training version")
        print("   2. Try model conversion script (see MODEL_CONVERSION.md)")
        print("   3. Retrain model with current TensorFlow version")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def get_risk_level(confidence):
    """Determine risk level based on confidence"""
    if confidence >= 0.90:
        return "high", "High Confidence"
    elif confidence >= 0.70:
        return "medium", "Medium Confidence"
    else:
        return "low", "Low Confidence - Retake Recommended"

def save_to_history(filename, predicted_class, confidence, timestamp):
    """Save prediction to history JSON file"""
    history_file = 'static/history/predictions.json'
    
    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new entry
    history.append({
        'filename': filename,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'timestamp': timestamp,
        'risk_level': get_risk_level(confidence)[1]
    })
    
    # Keep only last 50 predictions
    history = history[-50:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html', model_loaded=model is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess and predict
            img = preprocess_image(filepath)
            predictions = model.predict(img, verbose=0)
            
            # Get prediction results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            
            # Get risk level
            risk_level, risk_text = get_risk_level(confidence)
            
            # Get all class probabilities
            all_predictions = {
                CLASS_NAMES[i]: float(predictions[0][i]) 
                for i in range(len(CLASS_NAMES))
            }
            
            # Save to history
            save_to_history(filename, predicted_class, confidence, 
                          datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            return jsonify({
                'success': True,
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'risk_level': risk_level,
                'risk_text': risk_text,
                'all_predictions': all_predictions,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'}), 400

@app.route('/history')
def history():
    """View prediction history"""
    history_file = 'static/history/predictions.json'
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            predictions = json.load(f)
        predictions.reverse()  # Show newest first
    else:
        predictions = []
    
    return render_template('history.html', predictions=predictions)

@app.route('/about')
def about():
    """About page with model information"""
    model_info = {
        'model_name': 'MobileNetV3-Small',
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'classes': CLASS_NAMES,
        'model_loaded': model is not None
    }
    return render_template('about.html', info=model_info)

if __name__ == '__main__':
    print("="*50)
    print("🌱 Pumpkin Leaf Disease Detection System")
    print("="*50)
    
    # Load model on startup
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠️  WARNING: Model not loaded!")
        print("The application will start but predictions will not work.")
        print("Please place your trained model file in the 'models/' directory.")
    
    print("\n🚀 Starting Flask server...")
    print("📱 Access the app at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
