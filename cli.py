import argparse
from app.model import BaselineDiagnostics
from app.ai_client import generate_explanation

def main():
    """Run inference from the terminal."""
    parser = argparse.ArgumentParser(description='Run health diagnostics inference')
    parser.add_argument('--symptoms', required=True, help='Symptoms description')
    parser.add_argument('--age', type=int, default=30, help='Patient age')
    parser.add_argument('--sex', default='', help='Patient sex')
    parser.add_argument('--duration', type=int, default=1, help='Duration in days')
    parser.add_argument('--use-ai', action='store_true', help='Generate AI explanation')
    
    args = parser.parse_args()
    
    # Load model
    model = BaselineDiagnostics().load()
    
    # Prepare sample
    sample = {
        'symptoms': args.symptoms,
        'age': args.age,
        'sex': args.sex,
        'duration': args.duration
    }
    
    # Get predictions
    ranked = model.predict_proba(sample)
    
    print("Ranked conditions:")
    for i, condition in enumerate(ranked[:5]):
        print(f"{i+1}. {condition['label']} ({condition['prob']*100:.1f}%)")
    
    # Generate AI explanation if requested
    if args.use_ai:
        print("\nGenerating AI explanation...")
        explanation = generate_explanation(sample, ranked)
        if 'error' in explanation:
            print(f"Error: {explanation['error']}")
        else:
            print("\nAI Explanation:")
            print(explanation['explanation'])

if __name__ == '__main__':
    main()