
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import prune

# Set CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PyTorch Dataset wrapper
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x).to(device)
        self.y = torch.FloatTensor(y).to(device)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Dynamic Neural Network model using PyTorch components
class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_size, architecture):
        super(DynamicNeuralNetwork, self).__init__()
        
        # Unpack architecture parameters
        self.layer_sizes = architecture['layer_sizes']
        self.activations = architecture['activations']
        self.dropout_rates = architecture['dropout_rates']
        self.use_batch_norm = architecture['use_batch_norm']
        
        # Create layer containers using ModuleList for proper PyTorch tracking
        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        prev_size = input_size
        
        # Build network layers
        for i, (size, activation, dropout) in enumerate(zip(self.layer_sizes, self.activations, self.dropout_rates)):
            # Linear layer
            self.linear_layers.append(nn.Linear(prev_size, size))
            
            # Activation function
            if activation == 'relu':
                self.activation_layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.activation_layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.activation_layers.append(nn.Sigmoid())
            elif activation == 'leakyrelu':
                self.activation_layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                self.activation_layers.append(nn.ELU())
            
            # Batch normalization
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(size))
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            
            prev_size = size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Initialize weights using PyTorch's built-in initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for ReLU-based networks
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            x = self.activation_layers[i](x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.dropout_layers[i](x)
        
        return self.output_layer(x)

# PyTorch training process wrapped in a class
class NeuralNetTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, patience=50):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in self.train_loader:
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs.squeeze(), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        # Update learning rate based on validation performance
        self.scheduler.step(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, num_epochs):
        history = {'loss': [], 'val_loss': []}
        min_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            avg_train_loss = self.train_epoch()
            history['loss'].append(avg_train_loss)
            
            # Validate
            avg_val_loss = self.validate()
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model
        self.model.load_state_dict(best_model_state)
        
        return history, min_val_loss

# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def __init__(self, input_size, train_loader, val_loader, 
                 population_size=20, generations=10, 
                 mutation_rate=0.2, elite_size=2):
        self.input_size = input_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = []
        self.history = {'best_fitness': [], 'avg_fitness': []}
        
    def initialize_population(self):
        """Create initial random population of neural network architectures"""
        population = []
        
        for _ in range(self.population_size):
            # Random number of layers between 2-5
            num_layers = random.randint(2, 5)
            
            # Generate random architecture
            architecture = {
                'layer_sizes': [random.choice([64, 80, 96, 112, 128, 144, 160]) for _ in range(num_layers)],
                'activations': [random.choice(['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu']) for _ in range(num_layers)],
                'dropout_rates': [random.uniform(0.0, 0.5) for _ in range(num_layers)],
                'use_batch_norm': random.choice([True, False]),
                'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.005]),
                'batch_size': random.choice([16, 22, 32, 64]),
            }
            
            population.append(architecture)
        
        self.population = population
        return population
    
    def fitness(self, architecture):
        """Evaluate fitness of architecture by training and validating a model"""
        try:
            # Create model based on architecture
            model = DynamicNeuralNetwork(self.input_size, architecture).to(device)
            
            # Configure optimizer with weight decay (L2 regularization)
            optimizer = optim.Adam(model.parameters(), 
                                  lr=architecture['learning_rate'],
                                  weight_decay=0.0001)
            
            criterion = nn.L1Loss()  # MAE loss
            
            # Train for a few epochs to evaluate
            num_quick_epochs = 15  # Small number of epochs for quick evaluation
            model.train()
            
            for epoch in range(num_quick_epochs):
                for batch_x, batch_y in self.train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    outputs = model(batch_x)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Clear memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return negative loss as fitness (higher is better)
            return -avg_val_loss
        
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return float('-inf')
    
    def select_parents(self, population, fitnesses):
        """Select parents for breeding using tournament selection"""
        parents = []
        
        # First add the elites
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        for idx in elite_indices:
            parents.append(population[idx])
        
        # Tournament selection for the rest
        while len(parents) < self.population_size:
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Create a child architecture by combining aspects of two parents"""
        child = {}
        
        # Crossover layer structure
        layer_count = random.choice([len(parent1['layer_sizes']), len(parent2['layer_sizes'])])
        
        # Make sure we have enough layers to work with from both parents
        p1_layers = parent1['layer_sizes'][:layer_count] if len(parent1['layer_sizes']) >= layer_count else parent1['layer_sizes'] + [128] * (layer_count - len(parent1['layer_sizes']))
        p1_activations = parent1['activations'][:layer_count] if len(parent1['activations']) >= layer_count else parent1['activations'] + ['relu'] * (layer_count - len(parent1['activations']))
        p1_dropouts = parent1['dropout_rates'][:layer_count] if len(parent1['dropout_rates']) >= layer_count else parent1['dropout_rates'] + [0.1] * (layer_count - len(parent1['dropout_rates']))
        
        p2_layers = parent2['layer_sizes'][:layer_count] if len(parent2['layer_sizes']) >= layer_count else parent2['layer_sizes'] + [128] * (layer_count - len(parent2['layer_sizes']))
        p2_activations = parent2['activations'][:layer_count] if len(parent2['activations']) >= layer_count else parent2['activations'] + ['relu'] * (layer_count - len(parent2['activations']))
        p2_dropouts = parent2['dropout_rates'][:layer_count] if len(parent2['dropout_rates']) >= layer_count else parent2['dropout_rates'] + [0.1] * (layer_count - len(parent2['dropout_rates']))

        # Create child by taking elements from both parents
        child['layer_sizes'] = [random.choice([p1, p2]) for p1, p2 in zip(p1_layers, p2_layers)]
        child['activations'] = [random.choice([p1, p2]) for p1, p2 in zip(p1_activations, p2_activations)]
        child['dropout_rates'] = [random.choice([p1, p2]) for p1, p2 in zip(p1_dropouts, p2_dropouts)]
        
        # Other parameters
        child['use_batch_norm'] = random.choice([parent1['use_batch_norm'], parent2['use_batch_norm']])
        child['learning_rate'] = random.choice([parent1['learning_rate'], parent2['learning_rate']])
        child['batch_size'] = random.choice([parent1['batch_size'], parent2['batch_size']])
        
        return child
    
    def mutate(self, architecture):
        """Randomly mutate aspects of an architecture"""
        mutated = copy.deepcopy(architecture)
        
        # Potentially add or remove a layer
        if random.random() < self.mutation_rate:
            if len(mutated['layer_sizes']) > 2 and random.choice([True, False]):
                # Remove a random layer
                idx = random.randint(0, len(mutated['layer_sizes']) - 1)
                mutated['layer_sizes'].pop(idx)
                mutated['activations'].pop(idx)
                mutated['dropout_rates'].pop(idx)
            else:
                # Add a layer
                mutated['layer_sizes'].append(random.choice([64, 80, 96, 112, 128, 144, 160]))
                mutated['activations'].append(random.choice(['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu']))
                mutated['dropout_rates'].append(random.uniform(0.0, 0.5))
        
        # Mutate layer sizes
        for i in range(len(mutated['layer_sizes'])):
            if random.random() < self.mutation_rate:
                mutated['layer_sizes'][i] = random.choice([64, 80, 96, 112, 128, 144, 160])
        
        # Mutate activations
        for i in range(len(mutated['activations'])):
            if random.random() < self.mutation_rate:
                mutated['activations'][i] = random.choice(['relu', 'tanh', 'sigmoid', 'leakyrelu', 'elu'])
        
        # Mutate dropout rates
        for i in range(len(mutated['dropout_rates'])):
            if random.random() < self.mutation_rate:
                mutated['dropout_rates'][i] = random.uniform(0.0, 0.5)
        
        # Mutate other parameters
        if random.random() < self.mutation_rate:
            mutated['use_batch_norm'] = not mutated['use_batch_norm']
        
        if random.random() < self.mutation_rate:
            mutated['learning_rate'] = random.choice([0.0001, 0.0005, 0.001, 0.005])
        
        if random.random() < self.mutation_rate:
            mutated['batch_size'] = random.choice([16, 22, 32, 64])
        
        return mutated
    
    def evolve(self):
        """Run the genetic algorithm for the specified number of generations"""
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
            print(f"\nGeneration {generation+1}/{self.generations}")
            
            # Evaluate fitness
            fitnesses = []
            for i, architecture in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{len(self.population)}...")
                fitness = self.fitness(architecture)
                fitnesses.append(fitness)
                print(f"Architecture {i+1} fitness: {-fitness:.4f} (lower is better)")
            
            # Record statistics
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            avg_fitness = np.mean(fitnesses)
            best_architecture = self.population[best_idx]
            
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            
            print(f"Generation {generation+1} - Best Fitness: {-best_fitness:.4f}, Avg Fitness: {-avg_fitness:.4f}")
            print(f"Best architecture: {self.summarize_architecture(best_architecture)}")
            
            # Select parents
            parents = self.select_parents(self.population, fitnesses)
            
            # Create new population
            new_population = []
            
            # Add elites directly
            for i in range(self.elite_size):
                new_population.append(self.population[np.argsort(fitnesses)[-i-1]])
            
            # Create the rest through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        # Return the best architecture found
        final_fitnesses = [self.fitness(arch) for arch in self.population]
        best_idx = np.argmax(final_fitnesses)
        return self.population[best_idx], -final_fitnesses[best_idx]  # Return negative to show actual loss
    
    def summarize_architecture(self, architecture):
        """Generate a string summary of the architecture"""
        return {
            "layers": len(architecture['layer_sizes']),
            "sizes": architecture['layer_sizes'],
            "activations": architecture['activations'],
            "batch_norm": architecture['use_batch_norm'],
            "learning_rate": architecture['learning_rate'],
            "batch_size": architecture['batch_size']
        }

def train_best_model(best_architecture, input_size, X_train, y_train, X_val, y_val, num_epochs=1000):
    """Train the best model found by the genetic algorithm"""
    # Create datasets with new batch size
    batch_size = best_architecture['batch_size']
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with the best architecture
    model = DynamicNeuralNetwork(input_size, best_architecture).to(device)
    
    # Configure optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=best_architecture['learning_rate'],
        weight_decay=0.0001  # L2 regularization
    )
    
    criterion = nn.L1Loss()  # MAE loss
    
    # Create trainer
    trainer = NeuralNetTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        patience=50  # Early stopping patience
    )
    
    # Train the model
    history, min_val_loss = trainer.train(num_epochs)
    
    # Apply pruning to reduce model size
    print("Applying model pruning to reduce size...")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)  # Prune 20% of weights
    
    # Make pruning permanent
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    return model, history, min_val_loss

def plot_training_history(history):
    """Plot the training and validation loss curves"""
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
    plt.savefig('training_history.png')
    plt.show()

def plot_genetic_algorithm_history(ga_history):
    """Plot the genetic algorithm optimization history"""
    plt.figure(figsize=(10, 6))
    
    # Convert best_fitness to loss (negative)
    best_fitness_loss = [-x for x in ga_history['best_fitness']]
    avg_fitness_loss = [-x for x in ga_history['avg_fitness']]
    
    plt.plot(best_fitness_loss, label='Best Model Loss')
    plt.plot(avg_fitness_loss, label='Average Population Loss')
    
    plt.xlabel('Generation')
    plt.ylabel('Loss (MAE)')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.legend()
    plt.savefig('ga_optimization.png')
    plt.show()

def save_model_to_torchscript(model, input_size):
    """Save model in TorchScript format for deployment"""
    model.eval()
    # Create example input
    example_input = torch.randn(1, input_size).to(device)
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    # Save to file
    traced_model.save("model_torchscript.pt")
    print("Model saved in TorchScript format as model_torchscript.pt")

def main():
    # Set PyTorch to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load data
    dataframe = pd.read_csv("information.csv", delimiter=",", header=None)
    dataset = dataframe.values
    
    x_data = dataset[:, :-1]
    y_data = dataset[:, -1]
    
    # Split data
    train_ratio = 0.75  
    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1 - train_ratio, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Input features: {X_train.shape[1]}")
    
    # Create default tensors and loaders for GA evaluation
    batch_size = 32
    
    # Transfer data to tensors and device
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create TensorDatasets and DataLoaders using PyTorch's native solutions
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and run the genetic algorithm
    input_size = x_data.shape[1]
    print("Starting genetic algorithm optimization...")
    
    ga = GeneticAlgorithm(
        input_size=input_size,
        train_loader=train_loader,
        val_loader=val_loader,
        population_size=15,  
        generations=7,       
        mutation_rate=0.2,   
        elite_size=2         
    )
    
    best_architecture, best_loss = ga.evolve()
    
    print("\nGenetic Algorithm Optimization Complete!")
    print(f"Best architecture loss: {best_loss:.4f}")
    print(f"Best architecture: {ga.summarize_architecture(best_architecture)}")
    
    # Plot GA optimization history
    plot_genetic_algorithm_history(ga.history)
    
    # Train the best model fully
    print("\nTraining best model architecture...")
    best_model, history, final_val_loss = train_best_model(
        best_architecture, input_size, X_train, y_train, X_val, y_val
    )
    
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    best_model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        predictions = best_model(X_val_tensor).squeeze()
        mae = nn.L1Loss()(predictions, y_val_tensor).item()
        mse = nn.MSELoss()(predictions, y_val_tensor).item()
        
        print(f"Final evaluation - MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # Save model
    save_model = input("Do you want to save the model? (y/n): ")
    if save_model.lower() == 'y':
        # Save in both PyTorch and TorchScript formats
        torch.save(best_model.state_dict(), 'neural_network_ga_optimized.pth')
        
        # Save as TorchScript for deployment
        save_model_to_torchscript(best_model, input_size)
        
        # Also save the architecture configuration for future reference
        architecture_summary = ga.summarize_architecture(best_architecture)
        with open('best_architecture.txt', 'w') as f:
            f.write(str(architecture_summary))
        
        print("Model saved as neural_network_ga_optimized.pth")
        print("Architecture saved as best_architecture.txt")
    else:
        print("Model not saved.")

if __name__ == "__main__":
    main()
