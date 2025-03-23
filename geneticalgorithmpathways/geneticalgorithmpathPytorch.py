import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

class RegressionModel:
    def __init__(self, input_dim=1):
        self.input_dim = input_dim
        self.model = self.build_model()
        self.history = {'loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.MeanSquaredError(),
            metrics=['mse']
        )
        
        return model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=150,
                restore_best_weights=True
            )
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2
        ).history
        
        # Get the best validation loss from current training
        self.best_val_loss = np.min(self.history['val_loss'])
        return self.best_val_loss
    
    def predict(self, x):
        return self.model.predict(x, verbose=0).flatten()
    
    def evaluate(self, x, y):
        predictions = self.predict(x)
        mse = mean_squared_error(y, predictions)
        return mse
    
    def save(self, filename='model.keras'):
        self.model.save(filename)
        np.save('best_val_loss.npy', self.best_val_loss)
        print(f"Model saved to {filename} with validation loss: {self.best_val_loss:.4f}")
    
    def load(self, filename='model.keras'):
        if os.path.exists(filename):
            self.model = keras.models.load_model(filename)
            if os.path.exists('best_val_loss.npy'):
                self.best_val_loss = np.load('best_val_loss.npy')
            else:
                self.best_val_loss = float('inf')
            print(f"Previous model loaded from {filename}")
            print(f"Previous model's best validation loss: {self.best_val_loss:.4f}")
            return True
        return False

class OptimizedRegressionDataset:
    def __init__(self, x, y, scaling_factor=1.0, noise_level=0.0):
        self.x = x
        self.y = y
        self.scaling_factor = scaling_factor
        self.noise_level = noise_level
        self.history = {}
        
    def optimize(self, x_train, y_train, x_val, y_val, epochs=50):
        # Create new model
        current_model = RegressionModel(input_dim=self.x.shape[1])
        
        # Try to load previous model
        previous_model = RegressionModel(input_dim=self.x.shape[1])
        previous_model_exists = previous_model.load('regression_model.keras')
        
        if previous_model_exists:
            previous_val_loss = previous_model.evaluate(x_val, y_val)
            print(f"Previous model validation error: {previous_val_loss:.4f}")
        
        # Train current model
        current_val_loss = current_model.train(x_train, y_train, x_val, y_val, epochs=epochs)
        print(f"New model validation error: {current_val_loss:.4f}")
        
        # Save only if current model is better or no previous model exists
        if not previous_model_exists or current_val_loss < previous_val_loss:
            print(current_val_loss, " < ", previous_val_loss)
            print("New model performs better! Saving...")
            current_model.save('regression_model.keras')
            self.history = current_model.history
            return current_model
        else:
            print("Previous model performs better. Keeping previous model...")
            self.history = current_model.history
            return previous_model
        
    def get_optimized_data(self):
        x_optimized = self.x * self.scaling_factor + np.random.normal(0, self.noise_level, self.x.shape)
        return x_optimized, self.y

def main():
    # Example usage
    x_data = np.linspace(0, 10, 100).reshape(-1, 1)
    y_data = 2 * x_data + 1 + np.random.normal(0, 1, x_data.shape)
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    # Create dataset
    dataset = OptimizedRegressionDataset(x_train, y_train)
    
    # Optimize and train model
    model = dataset.optimize(x_train, y_train, x_val, y_val, epochs=5000)
    
    # Visualize training progress
    plt.figure(figsize=(10, 6))
    plt.plot(model.history['loss'], label='Training Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.axhline(y=model.best_val_loss, color='r', linestyle='--', 
                label=f'Best Validation Loss: {model.best_val_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()