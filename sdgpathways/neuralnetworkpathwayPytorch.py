import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom dataset class
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.dropout_value = 0.0695
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(self.dropout_value),
            
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(self.dropout_value),
            
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(self.dropout_value),
            
            nn.Linear(120, 120),
            nn.Tanh(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.06),
            
            nn.Linear(120, 120),
            nn.Sigmoid(),
            nn.BatchNorm1d(120),
            
            nn.Linear(120, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Load data - by Coralyn 
dataframe = pd.read_csv("information.csv", delimiter=",", header=None)
dataset = dataframe.values

x_data = dataset[:, :-1]
y_data = dataset[:, -1]

# Split data
train_ratio = 0.25  # Since original code uses test_size=0.75
X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1 - train_ratio)

print(type(X_train[0])) # <class 'numpy.ndarray'>


# Create datasets and dataloaders
train_dataset = RegressionDataset(X_train, y_train)
val_dataset = RegressionDataset(X_val, y_val)

batch_size = 22
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer
input_size = x_data.shape[1]
model = NeuralNetwork(input_size).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.L1Loss()  # MAE loss

# Early stopping parameters
early_stopping_patience = 50
min_val_loss = float('inf')
best_model = None
early_stopping_counter = 0

# Training loop
history = {'loss': [], 'val_loss': []}
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    history['loss'].append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_loss += criterion(outputs.squeeze(), batch_y).item()
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        best_model = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Find the minimum validation loss and its corresponding epoch
min_val_loss = min(history['val_loss'])
min_epoch = history['val_loss'].index(min_val_loss)

# Plot the training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')

# Add vertical line at minimum validation loss
plt.axvline(x=min_epoch, color='red', linestyle='--', label=f'Minimum Validation Loss at Epoch {min_epoch+1}')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save model
save_model = input("Do you want to save the model? (y/n): ")
if save_model.lower() == 'y':
    torch.save(best_model, 'neural_network.pth')
    print("Model saved as neural_network.pth")
else:
    print("Model not saved.")