import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, regularizers

class RegressionModel:
    def __init__(self, input_dim=1):
        self.input_dim = input_dim
        self.model = self.build_model()
        self.history = {'loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        
    def build_model(self):
        # Create a more complex model with:
        # - More layers and neurons
        # - Dropout for regularization
        # - Batch normalization for better training
        # - L2 regularization to prevent overfitting
        # - Different activation functions
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(128, activation='relu', input_shape=(self.input_dim,),
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers with increasing complexity
            layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1)
        ])
        
        # Use a fixed learning rate instead of a schedule to make it compatible with ReduceLROnPlateau
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),  # Fixed learning rate
            loss=losses.MeanSquaredError(),
            metrics=['mse', 'mae']  # Track both MSE and MAE
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def train(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
        # Enhanced callbacks for better training
        callbacks = [
            # Early stopping with more patience
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=200,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction when plateauing
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=50,
                min_lr=0.00001,
                verbose=1
            ),
            # Model checkpoint to save the best model
            keras.callbacks.ModelCheckpoint(
                'best_model_checkpoint.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
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
        r2 = r2_score(y, predictions)
        print(f"Evaluation - MSE: {mse:.4f}, R² Score: {r2:.4f}")
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
            print(f"New model ({current_val_loss:.4f}) performs better than previous model ({previous_val_loss:.4f if previous_model_exists else 'N/A'})")
            print("Saving new model...")
            current_model.save('regression_model.keras')
            self.history = current_model.history
            return current_model
        else:
            print(f"Previous model ({previous_val_loss:.4f}) performs better than new model ({current_val_loss:.4f})")
            print("Keeping previous model...")
            self.history = current_model.history
            return previous_model
        
    def get_optimized_data(self):
        x_optimized = self.x * self.scaling_factor + np.random.normal(0, self.noise_level, self.x.shape)
        return x_optimized, self.y

def load_csv_data(file_path):
    """
    Load data from a CSV file where the last column is the target value
    and all other columns are input features.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        x_data: Features (all columns except the last one)
        y_data: Target variable (last column)
    """
    print(f"Loading data from {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Display the first few rows to understand the data
        print("\nFirst 5 rows of the data:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values in the dataset:")
            print(missing_values[missing_values > 0])
            # Fill missing values or drop rows with missing values
            df = df.dropna()
            print(f"After dropping rows with missing values: {df.shape[0]} rows remaining")
        
        # The last column is the target variable, all others are features
        y_data = df.iloc[:, -1].values
        x_data = df.iloc[:, :-1].values
        
        print(f"\nFeature columns: {df.columns[:-1].tolist()}")
        print(f"Target column: {df.columns[-1]}")
        
        # Scale the features for better model performance
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)
        
        print(f"Prepared data: X shape: {x_data.shape}, Y shape: {y_data.shape}")
        return x_data, y_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def main():
    # Load data from CSV file
    csv_file_path = r"C:\Users\liche\OneDrive\Desktop\PycharmProjects\jadonsthingy\information.csv"
    x_data, y_data = load_csv_data(csv_file_path)
    
    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    # Create dataset
    dataset = OptimizedRegressionDataset(x_train, y_train)
    
    # Optimize and train model
    model = dataset.optimize(x_train, y_train, x_val, y_val, epochs=5000)
    
    # Visualize training progress
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(model.history['loss'], label='Training Loss')
    plt.plot(model.history['val_loss'], label='Validation Loss')
    plt.axhline(y=model.best_val_loss, color='r', linestyle='--', 
                label=f'Best Validation Loss: {model.best_val_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress on CSV Data')
    plt.legend()
    
    # Plot learning metrics if available
    if 'mae' in model.history and 'val_mae' in model.history:
        plt.subplot(2, 1, 2)
        plt.plot(model.history['mae'], label='Training MAE')
        plt.plot(model.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    # Evaluate on validation set
    val_mse = model.evaluate(x_val, y_val)
    print(f"Validation MSE: {val_mse:.4f}")
    
    # Make predictions
    y_pred = model.predict(x_val)
    
    # Calculate residuals
    residuals = y_val - y_pred
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    
    # Subplot 2: Residuals
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    # Subplot 3: Residual Histogram
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    # Subplot 4: Actual vs Predicted (line plot for time series if applicable)
    plt.subplot(2, 2, 4)
    indices = np.arange(len(y_val))
    plt.plot(indices, y_val, 'b-', label='Actual')
    plt.plot(indices, y_pred, 'r-', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()
    
if __name__ == "__main__":
    main()