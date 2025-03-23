
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from mefunctions import find_element

class ModelSelector:
    def __init__(self, root=None):
        self.model = None
        self.model_path = None
        self.validation_error = None
        
        # Create GUI if root is provided
        if root:
            self.root = root
            self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI interface"""
        self.root.title("Model Selector")
        self.root.geometry("600x400")
        
        # Model selection frame
        model_frame = tk.Frame(self.root, padx=10, pady=10)
        model_frame.pack(fill=tk.X)
        
        tk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W)
        self.path_var = tk.StringVar()
        tk.Entry(model_frame, textvariable=self.path_var, width=50).grid(row=0, column=1, padx=5)
        tk.Button(model_frame, text="Browse...", command=self.browse_model).grid(row=0, column=2, padx=5)
        tk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5)
        
        # Input frame
        input_frame = tk.Frame(self.root, padx=10, pady=10)
        input_frame.pack(fill=tk.X)
        
        tk.Label(input_frame, text="Input (chemical formula):").grid(row=0, column=0, sticky=tk.W)
        self.input_var = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.input_var, width=50).grid(row=0, column=1, padx=5)
        tk.Button(input_frame, text="Predict", command=self.make_prediction).grid(row=0, column=2, padx=5)
        
        # Results frame
        results_frame = tk.Frame(self.root, padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(results_frame, text="Results:").pack(anchor=tk.W)
        self.results_text = tk.Text(results_frame, height=15, width=70)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def browse_model(self):
        """Open file dialog to select model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras Models", "*.keras"), ("All Files", "*.*")]
        )
        if file_path:
            self.path_var.set(file_path)
            
    def load_model(self, model_path=None):
        """Load the selected model"""
        try:
            if model_path is None:
                model_path = self.path_var.get()
                
            if not model_path:
                if hasattr(self, 'root'):
                    messagebox.showerror("Error", "Please select a model file")
                return False
                
            if not os.path.exists(model_path):
                if hasattr(self, 'root'):
                    messagebox.showerror("Error", f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            
            # Load validation error if it exists
            error_path = model_path.replace('.keras', '_val_error.npy')
            if os.path.exists(error_path):
                self.validation_error = float(np.load(error_path))
                message = f"Model loaded successfully from {model_path}\nPrevious validation error: {self.validation_error:.4f}"
            else:
                message = f"Model loaded successfully from {model_path}\nNo previous validation error found"
                
            if hasattr(self, 'results_text'):
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, message)
            else:
                print(message)
                
            return True
                
        except Exception as e:
            error_message = f"Error loading model: {str(e)}"
            if hasattr(self, 'results_text'):
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, error_message)
                messagebox.showerror("Error", error_message)
            else:
                print(error_message)
            return False
            
    def prepare_input(self, input_data):
        """Prepare input data for prediction"""
        # If input is a chemical formula string, convert it using find_element
        if isinstance(input_data, str) and not input_data[0].isdigit():
            elements_list = find_element(input_data)
            # Convert to numpy array and reshape for model input
            return np.array(elements_list).reshape(1, -1)
        
        # If input is already a list (from find_element), convert to numpy array
        elif isinstance(input_data, list):
            return np.array(input_data).reshape(1, -1)
            
        # If input is a space-separated string of numbers
        elif isinstance(input_data, str):
            try:
                numbers = list(map(float, input_data.split()))
                return np.array(numbers).reshape(1, -1)
            except ValueError:
                raise ValueError("Input string must contain valid numbers")
                
        else:
            raise ValueError("Unsupported input type")
            
    def make_prediction(self):
        """Process input and make prediction"""
        if self.model is None:
            message = "Please load a model first"
            if hasattr(self, 'results_text'):
                messagebox.showerror("Error", message)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, message)
            else:
                print(message)
            return
            
        try:
            # Get input from GUI or use provided input
            if hasattr(self, 'input_var'):
                input_data = self.input_var.get()
            else:
                input_data = input("Enter chemical formula or space-separated numbers: ")
                
            # Prepare input data
            input_array = self.prepare_input(input_data)
            
            # Check if input shape matches model's expected input shape
            expected_shape = self.model.input_shape[1]
            actual_shape = input_array.shape[1]
            
            if actual_shape != expected_shape:
                # Try to pad or truncate to match expected shape
                if actual_shape < expected_shape:
                    # Pad with zeros
                    padded = np.zeros((1, expected_shape))
                    padded[0, :actual_shape] = input_array[0, :]
                    input_array = padded
                else:
                    # Truncate
                    input_array = input_array[:, :expected_shape]
                    
            # Make prediction
            prediction = self.model.predict(input_array, verbose=0)
            
            # Format and display results
            result_message = f"Input: {input_data}\n"
            if self.validation_error is not None:
                result_message += f"Model validation error: {self.validation_error:.4f}\n"
            result_message += f"Prediction:\n{prediction}"
            
            if hasattr(self, 'results_text'):
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, result_message)
            else:
                print(result_message)
                
            return prediction
                
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            if hasattr(self, 'results_text'):
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, error_message)
                messagebox.showerror("Error", error_message)
            else:
                print(error_message)
            return None

def main():
    """Run the model selector GUI"""
    root = tk.Tk()
    app = ModelSelector(root)
    root.mainloop()

def predict_with_model(model_path, input_data):
    """Command-line function to make predictions with a model"""
    selector = ModelSelector()
    if selector.load_model(model_path):
        return selector.make_prediction(input_data)
    return None

if __name__ == "__main__":
    main()
