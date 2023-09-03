from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class EventClassification:
    def __init__(self, num_estimators=100) -> None:
        self.num_estimators = num_estimators
        self.model = self.init_model()

    def init_model(self):
        return RandomForestClassifier(n_estimators=self.num_estimators)
    
    def train(self, train_feats, train_labels):
        self.classifier = self.model.fit(train_feats, train_labels)

    def test(self, data):
        return self.model.predict(data)
    
    def load_model(self, model_path):
        self.classifier = joblib.load(model_path)

    def save_model(self, model_path):
        joblib.dump(self.classifier, model_path)
        
    def get_accuracy_score(self, predictions, ground_truth):
        return accuracy_score(ground_truth, predictions)
    
    def get_confusion_matrix(self, predictions, ground_truth):
        return confusion_matrix(ground_truth,predictions)