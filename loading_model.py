import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mefunctions import find_element

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.validation_error = None
        self.load_model()

    def load_model(self):
        """Load the saved Keras model and its validation error if available"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")

        print(self.model.summary())
        
        # Load validation error if it exists
        error_path = self.model_path.replace('.keras', '_val_error.npy')
        if os.path.exists(error_path):
            self.validation_error = float(np.load(error_path))
            print(f"Previous model validation error: {self.validation_error:.4f}")
        else:
            print("No previous validation error found")

    def prepare_input(self, input_string): 
        """Convert space-separated string input to numpy array""" 
        try:
            numbers = list(map(int, input_string.split()))
            return np.array(numbers).reshape(1, 120)
        except ValueError as e:
            raise ValueError("Input string must contain valid integers") from e
        except Exception as e:
            raise Exception("Error preparing input: " + str(e)) from e

    def predict(self, input_string):
        """Make prediction and show validation error"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        if self.validation_error is not None:
            print(f"Previous model validation error: {self.validation_error:.4f}")
        
        input_array = self.prepare_input(input_string)
        prediction = self.model.predict(input_array, verbose=0)
        return prediction

    def evaluate_with_labels(self, input_array, true_labels):
        """Evaluate model and save validation error"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        try:
            error = self.model.evaluate(input_array, true_labels, verbose=0)
            error_path = self.model_path.replace('.keras', '_val_error.npy')
            np.save(error_path, error[0])
            self.validation_error = error[0]
            return error
        except Exception as e:
            raise Exception("Error during evaluation: " + str(e)) from e

def main():
    """Example usage of ModelLoader"""
    try:
        # Initialize model loader
        model_path = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\jadonsthingy\Models\my_model.keras"
        loader = ModelLoader(model_path)

        # Example input
        input_string = "SiCl4"

        # Make prediction
        prediction = loader.predict(find_element(input_string))
        print(f"Model prediction: {prediction}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()