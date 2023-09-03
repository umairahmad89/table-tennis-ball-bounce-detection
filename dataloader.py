import json
from typing import Any
import numpy as np

class DataLoader:
    def __init__(self, json_path) -> None:
        self.path = json_path

    def load_data(self):
        f = open(self.path, "r")
        self.data = json.load(f)
    
    def preprocessing(self):
        '''
        Args:
            data: A list of lists, where each inner list contains ball coordinates in consecutive 9 frames.
                [-1, -1] represents a missing value.
        Return:
            A list containing only the y-coordinates of the ball in each frame.
        '''
        processed_data = []
        for sequence in self.data:
            # Extract y-coordinates and other features
            y_coordinates = [point[1] for point in sequence[:-5]]
            other_features = sequence[-1]  # Excluding the label
            # Calculate changes in y-coordinate
            y_changes = [y - y_coordinates[i - 1] for i, y in enumerate(y_coordinates)][1:]
            # Create a new entry with y-coordinate changes and other features
            new_entry = y_changes + other_features
            # Add the new entry to the new_data list
            processed_data.append(new_entry)
            new_entry = []

        return processed_data
    
    def get_data(self):
        self.load_data()
        processed_data = self.preprocessing()
        return processed_data
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        processed_data = self.get_data()
        features = [entry[:8] for entry in processed_data]
        labels = [entry[-1] for entry in processed_data]
        return features, labels 