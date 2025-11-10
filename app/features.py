import re

# A list of known symptom tokens
COMMON_SYMPTOMS = [
    'fever', 'cough', 'fatigue', 'sore throat', 'headache', 'nausea', 'vomiting',
    'diarrhea', 'abdominal pain', 'joint pain', 'stiffness', 'swelling',
    'chest pain', 'shortness of breath', 'sneezing', 'runny nose', 'itchy eyes',
    'rash', 'itching', 'redness', 'burning', 'urination', 'chills', 'body aches',
    'loss of taste', 'sensitivity to light', 'stiff neck', 'confusion',
    'blue lips', 'unconscious', 'bleeding', 'dizziness', 'congestion'
]

def normalize_symptom_token(token):
    """Lowercases, trims, and cleans punctuation."""
    return re.sub(r'[^\w\s]', '', token.lower().strip())

def vectorize_symptoms(symptoms_text):
    """Converts comma-separated symptoms into a bag-of-words vector over COMMON_SYMPTOMS."""
    if not symptoms_text:
        return [0] * len(COMMON_SYMPTOMS)
    
    symptoms_list = [normalize_symptom_token(s) for s in symptoms_text.split(',')]
    vector = []
    for symptom in COMMON_SYMPTOMS:
        # Check if symptom or a variation is in the symptoms list
        found = any(symptom in s or s in symptom for s in symptoms_list)
        vector.append(1 if found else 0)
    return vector

def encode_sex(sex):
    """One-hot encodes sex into (male, female) flags."""
    male = 1 if sex and sex.lower() in ['male', 'm'] else 0
    female = 1 if sex and sex.lower() in ['female', 'f'] else 0
    return [male, female]