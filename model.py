import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class BaselineDiagnostics:
    def __init__(self):
        self.pipeline = None
        self.labels = None
    
    def prepare_xy(self, df):
        """Converts dataset rows into feature arrays and label targets."""
        from app.features import vectorize_symptoms, encode_sex
        
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Extract features
            symptom_vector = vectorize_symptoms(row['symptoms'])
            age = float(row.get('age', 0)) if row.get('age') else 0
            sex_encoded = encode_sex(row.get('sex', ''))
            duration = float(row.get('duration', 0)) if row.get('duration') else 0
            
            # Combine all features
            features = symptom_vector + [age] + sex_encoded + [duration]
            X.append(features)
            y.append(row['condition'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df):
        """Trains a logistic regression model inside a pipeline with a StandardScaler."""
        X, y = self.prepare_xy(df)
        
        # Build pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=200))
        ])
        
        # Train model
        self.pipeline.fit(X, y)
        
        # Store labels
        self.labels = list(set(y))
        
        return self
    
    def evaluate(self, df):
        """Returns a classification_report on the test set."""
        from sklearn.metrics import classification_report
        X, y = self.prepare_xy(df)
        y_pred = self.pipeline.predict(X)
        return classification_report(y, y_pred)
    
    def predict_proba(self, sample):
        """Produces a ranked list of {label, prob} for a given input sample."""
        from app.features import vectorize_symptoms, encode_sex
        
        if not self.pipeline or not self.labels:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Extract features from sample
        symptom_vector = vectorize_symptoms(sample.get('symptoms', ''))
        age = float(sample.get('age', 0))
        sex_encoded = encode_sex(sample.get('sex', ''))
        duration = float(sample.get('duration', 0))
        
        # Combine all features
        features = symptom_vector + [age] + sex_encoded + [duration]
        X = np.array([features])
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(X)[0]
        
        # Create ranked list
        ranked = []
        for i, label in enumerate(self.pipeline.classes_):
            ranked.append({
                'label': label,
                'prob': float(probabilities[i])
            })
        
        # Sort by probability (descending)
        ranked.sort(key=lambda x: x['prob'], reverse=True)
        
        return ranked
    
    def save(self, model_path='model/diagnostics_model.joblib', labels_path='model/labels.joblib'):
        """Persists model and labels via joblib."""
        import os
        os.makedirs('model', exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.labels, labels_path)
    
    def load(self, model_path='model/diagnostics_model.joblib', labels_path='model/labels.joblib'):
        """Loads model and labels via joblib."""
        self.pipeline = joblib.load(model_path)
        self.labels = joblib.load(labels_path)
        return self