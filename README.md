# Quantum-Classical ML Systems

> A hybrid machine learning demonstration platform showcasing production-ready classical models alongside research-phase quantum computing capabilities for venture capital and technical presentations.

## Overview

This project demonstrates a systematic approach to emerging quantum computing technologies by implementing a comparative analysis framework between classical machine learning algorithms and quantum variational circuits. The platform features a professional web interface for real-time model inference and performance visualization.

### Key Features

- **Production-Ready Classical Models**: Logistic Regression, Support Vector Machines, and Random Forest classifiers optimized for deployment
- **Quantum Computing Research**: Variational Quantum Classifier (VQC) implementation demonstrating quantum readiness
- **Interactive Web Interface**: High-tech UI with real-time predictions and performance metrics
- **Transparent Performance Reporting**: Honest accuracy metrics without inflated claims
- **RESTful API Architecture**: Flask-based backend for easy integration and scalability

## Live Demo

![Demo Screenshot to be added]

**Features Showcase:**
- Real-time multi-model predictions
- Performance comparison dashboard
- Accuracy visualization with interactive charts
- Processing time metrics for each model

## Model Performance

| Model | Accuracy | Training Time | Status |
|-------|----------|---------------|---------|
| Random Forest | ~91% | 2.1s | Production Ready |
| Logistic Regression | ~87% | 0.23s | Production Ready |
| Support Vector Machine | ~84% | 1.45s | Production Ready |
| Quantum VQC | ~73% | 127.3s | Research Phase |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Dashboard  │  │  Prediction  │  │  Metrics     │  │
│  │  Interface  │  │  Interface   │  │  Visualization│  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────┐
│                 Flask Backend (Python)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model     │  │ Preprocessing│  │   Inference  │  │
│  │   Loader    │  │   Pipeline   │  │   Engine     │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Pre-trained Models Storage                 │
│  • logistic_regression.joblib                          │
│  • svm.joblib                                          │
│  • random_forest.joblib                                │
│  • best_quantum_model.pkl                              │
│  • preprocessing.pkl                                   │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack

**Backend:**
- Python 3.8+
- Flask (Web Framework)
- scikit-learn (Classical ML Models)
- Qiskit (Quantum Computing)
- NumPy (Numerical Computing)
- joblib (Model Serialization)

**Frontend:**
- HTML5/CSS3
- Vanilla JavaScript
- Chart.js (Data Visualization)
- Modern CSS (Glassmorphism, Gradients)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## NOTE: The easiest way to access the interface containing the model information is to download 'demo.html'. This will take you to a static interface, however, for a more dynamic interface, follow the setup instructions below.
### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Healora/quantum-prediction-project.git
   cd quantum-prediction-project
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors scikit-learn numpy joblib qiskit
   ```

4. **Verify model files**
   
   Ensure the `extracted_models/` folder contains:
   - `logistic_regression.joblib`
   - `svm.joblib`
   - `random_forest.joblib`
   - `best_quantum_model.pkl`
   - `preprocessing.pkl`
   - `extraction_sumamry.json`

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the interface**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Making Predictions

1. **Navigate to the Live Prediction Interface** section
2. **Input four numerical features** (Feature 1-4)
3. **Click "Generate Predictions"**
4. **View results** from all loaded models with confidence scores and processing times

### API Endpoints

#### Health Check
```http
GET /health
```
Returns server status and loaded models.

#### Model Information
```http
GET /model-info
```
Returns detailed information about all loaded models.

#### Make Predictions
```http
POST /predict
Content-Type: application/json

{
  "feature1": 0.5,
  "feature2": 0.3,
  "feature3": 0.8,
  "feature4": 0.2
}
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "logistic": {
      "prediction": "Class A",
      "confidence": 0.82,
      "inference_time": 2.1
    },
    "svm": {...},
    "randomforest": {...}
  }
}
```

## Model Training

The models in this repository were trained using the following methodology:

1. **Data Preprocessing**: StandardScaler normalization
2. **Train-Test Split**: 80-20 ratio
3. **Hyperparameter Optimization**: Grid search with cross-validation
4. **Quantum Circuit**: 4-qubit variational circuit with RealAmplitudes ansatz

For model retraining, refer to the Jupyter notebooks in the `notebooks/` directory.

## Project Structure

```
quantum-prediction-project/
├── app.py                      # Flask backend server
├── index.html                  # Frontend interface
├── extracted_models/           # Pre-trained model storage
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── random_forest.joblib
│   ├── best_quantum_model.pkl
│   └── preprocessing.pkl
├── notebooks/                  # Training notebooks (optional)
│   └── model_training.ipynb
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # Project license
```

## Research Context

This project was developed as part of research into hybrid quantum-classical machine learning systems. Key findings:

- **Classical Superiority**: Current classical models significantly outperform quantum approaches for this problem domain
- **Quantum Potential**: VQC demonstrates feasibility but requires further optimization
- **Practical Insights**: Training time and inference speed remain critical challenges for quantum ML
- **Future Directions**: Research continues into quantum advantage for specific problem classes

## License

This project is licensed under the MIT License.

## Authors

**Safura Kasu**
- GitHub: https://github.com/SafuraKasu
- LinkedIn: https://www.linkedin.com/in/safura-kasu/

## Acknowledgments

- Qiskit team for quantum computing framework
- scikit-learn community for classical ML tools
- Flask community for web framework

## Contact

For questions, collaborations, or venture capital inquiries:

- **Email**: healora98@gmail.com
- **Website**: https://healora.org/

---

**Note**: This is a research demonstration project. Classical models are production-ready, while quantum models represent ongoing research into quantum machine learning capabilities.

## Roadmap

- [ ] Implement additional quantum circuits (QAOA, QNN)
- [ ] Add model explainability features (SHAP, LIME)
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] Create Docker containerization
- [ ] Add comprehensive test suite
- [ ] Implement model retraining pipeline
- [ ] Add user authentication and session management
- [ ] Create detailed API documentation with Swagger

---

⭐ **If you find this project useful, please consider giving it a star!**
