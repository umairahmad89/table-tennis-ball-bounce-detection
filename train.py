from model import EventClassification
from dataloader import DataLoader
import argparse
from sklearn.model_selection import train_test_split

def main(args):
        # load data
    # split train and test
    # show accuracy and confusion matrix on test data
    # Access the parsed arguments
    json_path = args.json_path
    num_estimators = args.num_estimators
    train_size = args.train_size
    save_path = args.save_path

    dataloader = DataLoader(json_path)
    features, labels = dataloader()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size, random_state=42)

    # now init model
    model = EventClassification(num_estimators)
    model.train(X_train, y_train)
    predictions = model.test(X_test)
    print("Accuracy Test: ",model.get_accuracy_score(predictions, y_test))
    print("Confusion Matrix Test: ",model.get_confusion_matrix(predictions, y_test))
    model.save_model(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BounceClassifier to \
                                     classify bounce and not bounce events of table tennis game...")
    # Add arguments
    parser.add_argument("--json_path", type=str, default=None, help="Path to the JSON file")
    parser.add_argument("--num_estimators", type=int, default=100, help="Number of estimators (default: 100)")
    parser.add_argument("--train_size", type=float, default=0.8, help="Training data size (default: 0.8)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the output")
    # Parse arguments
    args = parser.parse_args()

    main(args)
