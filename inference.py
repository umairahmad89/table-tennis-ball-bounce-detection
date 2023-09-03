from model import EventClassification
from dataloader import DataLoader
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inference script for EventClassification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (RF model)")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing data for inference")
    
    args = parser.parse_args()
    
    # Load the trained model
    model = joblib.load(args.model_path)

    # Create a DataLoader instance
    dataloader = DataLoader(args.json_path)

    # Load and preprocess the data
    features, labels = dataloader()

    # Perform inference using the loaded model
    predictions = model.test(features)

    # Display the results
    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: Predicted Class - {prediction}")

if __name__ == "__main__":
    main()
