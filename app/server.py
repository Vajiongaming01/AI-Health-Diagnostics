import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from app.model import BaselineDiagnostics
from app.ai_client import generate_explanation

app = Flask(__name__)
CORS(app)

# Load the trained model
model = BaselineDiagnostics()
try:
    model.load()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Parse request data
        data = request.get_json(force=True)
        symptoms = data.get('symptoms', '')
        age = data.get('age', 0)
        sex = data.get('sex', '')
        duration = data.get('duration', 0)
        history = data.get('history', '')
        notes = data.get('notes', '')
        
        if not symptoms.strip():
            return jsonify({"error": "Symptoms are required"}), 400
        
        # Prepare sample for prediction
        sample = {
            'symptoms': symptoms,
            'age': age,
            'sex': sex,
            'duration': duration
        }
        
        # Get ML predictions
        ranked_conditions = model.predict_proba(sample)
        
        # Generate AI explanation
        explanation_data = generate_explanation({
            'symptoms': symptoms,
            'age': age,
            'sex': sex,
            'duration': duration,
            'history': history,
            'notes': notes
        }, ranked_conditions)
        
        # Prepare response
        response_data = {
            'ranked': ranked_conditions,
            'explanation': explanation_data
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/<path:path>')
def static_proxy(path):
    # Serve static files
    if os.path.exists(path):
        return send_from_directory('.', path)
    return 'Not found', 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)