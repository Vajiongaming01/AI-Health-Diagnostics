import pandas as pd
from sklearn.model_selection import train_test_split
from app.data import load_demo
from app.model import BaselineDiagnostics

def main():
    """Entry point to train the demo model using the demo dataset."""
    # Load demo data
    df = load_demo()
    print(f"Loaded {len(df)} training samples")
    
    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    
    # Train model
    model = BaselineDiagnostics()
    model.fit(train_df)
    
    # Evaluate on test set
    print("\nEvaluation on test set:")
    report = model.evaluate(test_df)
    print(report)
    
    # Save model
    model.save()
    print("\nModel saved to model/ directory")

if __name__ == '__main__':
    main()