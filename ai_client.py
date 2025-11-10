import os
import requests

def generate_explanation(payload, ranked_conditions):
    """Calls an AI provider to generate an explanation based on the ML results."""
    # Read API key from environment
    api_key = os.environ.get('DIAG_AI_API_KEY')
    
    if not api_key:
        return {
            "error": "AI explanation is disabled. Set DIAG_AI_API_KEY environment variable to enable."
        }
    
    # Get endpoint and model from environment or use defaults
    endpoint = os.environ.get('DIAG_AI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
    model = os.environ.get('DIAG_AI_MODEL', 'gpt-3.5-turbo')
    
    # Format prompt
    prompt = format_prompt(payload, ranked_conditions)
    
    # Prepare headers and data
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are a clinical decision support AI. You are not a doctor. Provide helpful, cautious, evidence-informed guidance.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7
    }
    
    try:
        # Make API request
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        explanation = result['choices'][0]['message']['content']
        
        return {
            'explanation': explanation
        }
    except Exception as e:
        return {
            'error': f'Failed to generate explanation: {str(e)}'
        }

def format_prompt(payload, ranked_conditions):
    """Builds a clear prompt containing user input and local ML probabilities."""
    lines = []
    lines.append("Based on the following symptom analysis, provide a detailed explanation:")
    lines.append("")
    
    # Add ranked conditions
    lines.append("Most likely conditions:")
    for i, condition in enumerate(ranked_conditions[:5]):
        lines.append(f"{i+1}. {condition['label']} ({condition['prob']*100:.1f}%)")
    
    lines.append("")
    lines.append("Patient information:")
    for key, value in payload.items():
        if value:
            lines.append(f"- {key}: {value}")
    
    lines.append("")
    lines.append("Structure your response with the following sections:")
    lines.append("1) Probable conditions (top 3-5) with brief rationale")
    lines.append("2) Severity assessment (mild/moderate/severe) with reasoning and red flags")
    lines.append("3) What to do now: step-by-step self-care and when to seek urgent care")
    lines.append("4) How long it may last and expected progression")
    lines.append("5) What to tell a doctor and suggested tests to discuss")
    lines.append("6) Clear disclaimer that you are not a medical professional")
    lines.append("")
    lines.append("Be concise but thorough. Avoid overconfident statements.")
    
    return "\n".join(lines)