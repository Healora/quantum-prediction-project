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
    
    print("🔍 Looking for models in 'extracted_models' folder...")
    
    # Find the extracted_models folder
    models_folder = 'extracted_models'
    if not os.path.exists(models_folder):
        print("❌ 'extracted_models' folder not found!")
        return False
    
    try:
        # Load preprocessing scaler
        scaler_file = os.path.join(models_folder, 'preprocessing.pkl')
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            print(f"✅ Loaded preprocessing scaler")
        
        # Load Logistic Regression
        lr_file = os.path.join(models_folder, 'logistic_regression.joblib')
        if os.path.exists(lr_file):
            models['logistic'] = joblib.load(lr_file)
            print(f"✅ Loaded Logistic Regression")
        
        # Load SVM
        svm_file = os.path.join(models_folder, 'svm.joblib')
        if os.path.exists(svm_file):
            models['svm'] = joblib.load(svm_file)
            print(f"✅ Loaded SVM")
        
        # Load Random Forest
        rf_file = os.path.join(models_folder, 'random_forest.joblib')
        if os.path.exists(rf_file):
            models['randomforest'] = joblib.load(rf_file)
            print(f"✅ Loaded Random Forest")
        
        # Load Quantum model (optional - might fail)
        quantum_file = os.path.join(models_folder, 'best_quantum_model.pkl')
        if os.path.exists(quantum_file):
            try:
                with open(quantum_file, 'rb') as f:
                    models['quantum'] = pickle.load(f)
                print(f"✅ Loaded Quantum model")
            except Exception as e:
                print(f"⚠️  Quantum model failed to load: {str(e)}")
                print(f"⚠️  Continuing without quantum model...")
        
        # Load metadata if available
        metadata_files = glob.glob(os.path.join(models_folder, 'extraction_summary_*.json'))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                model_metadata = json.load(f)
            print(f"✅ Loaded metadata: {os.path.basename(metadata_files[0])}")
        
        # Check if at least some models loaded
        if len(models) == 0:
            print("❌ No models were loaded!")
            return False
        
        print(f"\n🎉 Successfully loaded {len(models)} models!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
    """Serve the main interface"""
    from flask import render_template
    return render_template('index.html')

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