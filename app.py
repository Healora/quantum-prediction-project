from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import os
import glob
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # This allows your HTML to talk to this server

# Global variables to store loaded models
models = {}
scaler = None
model_metadata = {}

def load_models():
    """Load all models from the extracted_models folder"""
    global models, scaler, model_metadata
    
    print("Looking for models in 'extracted_models' folder...")
    
    # Find the extracted_models folder
    models_folder = 'extracted_models'
    if not os.path.exists(models_folder):
        print("'extracted_models' folder not found!")
        return False
    
    try:
        # Load preprocessing scaler
        scaler_files = glob.glob(os.path.join(models_folder, 'preprocessing_*.pkl'))
        if scaler_files:
            scaler = joblib.load(scaler_files[0])
            print(f"Loaded scaler: {os.path.basename(scaler_files[0])}")
        
        # Load Logistic Regression
        lr_files = glob.glob(os.path.join(models_folder, 'logistic_regression_*.joblib'))
        if lr_files:
            models['logistic'] = joblib.load(lr_files[0])
            print(f"Loaded Logistic Regression: {os.path.basename(lr_files[0])}")
        
        # Load SVM
        svm_files = glob.glob(os.path.join(models_folder, 'svm_*.joblib'))
        if svm_files:
            models['svm'] = joblib.load(svm_files[0])
            print(f"Loaded SVM: {os.path.basename(svm_files[0])}")
        
        # Load Random Forest
        rf_files = glob.glob(os.path.join(models_folder, 'random_forest_*.joblib'))
        if rf_files:
            models['randomforest'] = joblib.load(rf_files[0])
            print(f"Loaded Random Forest: {os.path.basename(rf_files[0])}")
        
        # Load Quantum model
        quantum_files = glob.glob(os.path.join(models_folder, 'best_quantum_model_*.pkl'))
        if quantum_files:
            with open(quantum_files[0], 'rb') as f:
                models['quantum'] = pickle.load(f)
            print(f"Loaded Quantum model: {os.path.basename(quantum_files[0])}")
        
        # Load metadata if available
        metadata_files = glob.glob(os.path.join(models_folder, 'extraction_summary_*.json'))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                model_metadata = json.load(f)
            print(f"Loaded metadata: {os.path.basename(metadata_files[0])}")
        
        print(f"\nSuccessfully loaded {len(models)} models!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Check if server is running and models are loaded"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    info = {
        'models': {}
    }
    
    for key, model in models.items():
        model_info = {
            'name': model.__class__.__name__,
            'loaded': True
        }
        
        # Add metadata if available
        if key in model_metadata:
            model_info.update(model_metadata[key])
        
        info['models'][key] = model_info
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with all loaded models"""
    try:
        # Get the input data
        data = request.json
        features = [
            float(data['feature1']),
            float(data['feature2']),
            float(data['feature3']),
            float(data['feature4'])
        ]
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Apply preprocessing if scaler is available
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make predictions with each model
        predictions = {}
        
        for model_key, model in models.items():
            start_time = datetime.now()
            
            try:
                # Get prediction
                prediction = model.predict(X)[0]
                
                # Get probability if available
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    confidence = float(max(proba))
                
                # Calculate inference time
                inference_time = (datetime.now() - start_time).total_seconds() * 1000  # in ms
                
                predictions[model_key] = {
                    'prediction': f'Class {prediction}',
                    'confidence': confidence,
                    'inference_time': round(inference_time, 2)
                }
                
            except Exception as e:
                predictions[model_key] = {
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'input_features': features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/')
def home():
    """Simple home page"""
    return """
    <h1>ML Model Server is Running!</h1>
    <p>Models loaded: {}</p>
    <p>Available endpoints:</p>
    <ul>
        <li>GET /health - Check server status</li>
        <li>GET /model-info - Get model information</li>
        <li>POST /predict - Make predictions</li>
    </ul>
    """.format(', '.join(models.keys()))

if __name__ == '__main__':
    print("=" * 50)
    print("Starting ML Model Server...")
    print("=" * 50)
    
    # Load models on startup
    if load_models():
        print("\n" + "=" * 50)
        print("Server ready!")
        print("Open your browser to: http://localhost:5000")
        print("=" * 50 + "\n")
        
        # Start the server
        app.run(debug=True, port=5000)
    else:
        print("\nFailed to load models. Please check your 'extracted_models' folder.")