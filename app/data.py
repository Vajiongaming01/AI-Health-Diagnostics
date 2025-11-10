import pandas as pd
import os

def ensure_demo_dataset():
    """Creates a CSV with basic rows that map symptoms+metadata to example labels."""
    data_path = os.path.join('data', 'sample_training.csv')
    
    if not os.path.exists(data_path):
        os.makedirs('data', exist_ok=True)
        # Seed dataset
        seed = [
            {"symptoms": "fever cough fatigue sore throat", "condition": "Common Cold"},
            {"symptoms": "fever dry cough loss of taste shortness of breath", "condition": "COVID-19"},
            {"symptoms": "sneezing runny nose itchy eyes", "condition": "Allergic Rhinitis"},
            {"symptoms": "nausea vomiting diarrhea abdominal pain", "condition": "Gastroenteritis"},
            {"symptoms": "headache sensitivity to light nausea", "condition": "Migraine"},
            {"symptoms": "burning urination frequent urination lower abdominal pain", "condition": "Urinary Tract Infection"},
            {"symptoms": "joint pain stiffness swelling", "condition": "Arthritis"},
            {"symptoms": "chest pain shortness of breath sweating nausea", "condition": "Possible Cardiac Issue"},
            {"symptoms": "fever chills body aches cough", "condition": "Influenza"},
            {"symptoms": "rash itching redness swelling", "condition": "Dermatitis"},
        ]
        pd.DataFrame(seed).to_csv(data_path, index=False)
    return data_path

def load_demo():
    """Returns a pandas DataFrame containing the demo data."""
    data_path = ensure_demo_dataset()
    return pd.read_csv(data_path)