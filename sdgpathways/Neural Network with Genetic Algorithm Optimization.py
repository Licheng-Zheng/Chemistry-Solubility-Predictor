
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any
import copy

# Set CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class
class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Dynamic Neural Network model
class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_size, architecture):
        super(DynamicNeuralNetwork, self).__init__()
        
        # Unpack architecture parameters
        self.layer_sizes = architecture['layer_sizes']
        self.activations = architecture['activations']
        self.dropout_rates = architecture['dropout_rates']
        self.use_batch_norm = architecture['use_batch_norm']
        
        # Build dynamic layers
        layers = []
        prev_size = input_size
        
        for i, (size, activation, dropout) in enumerate(zip(self.layer_sizes, self.activations, self.dropout_rates)):
            # Add linear layer
            layers.append(nn.Linear(prev_size, size))
            
            # Add activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            # Add batch normalization if specified
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            
            # Add dropout if rate > 0
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = size
        
        # Add final output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

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
                # print("matthew was here")
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
            
            # Configure optimizer
            optimizer = optim.Adam(model.parameters(), lr=architecture['learning_rate'])
            criterion = nn.L1Loss()  # MAE loss
            
            # Train for a few epochs to evaluate
            num_quick_epochs = 15  # Small number of epochs for quick evaluation
            model.train()
            
            for epoch in range(num_quick_epochs):
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
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
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Return negative loss as fitness (higher is better)
            return -avg_val_loss
        
        except Exception as e:
            # /kill @e Karanveer 
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
    # Create datasets with possibly new batch size
    batch_size = best_architecture['batch_size']
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with the best architecture
    model = DynamicNeuralNetwork(input_size, best_architecture).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_architecture['learning_rate'])
    criterion = nn.L1Loss()  # MAE loss
    
    # Early stopping parameters
    early_stopping_patience = 50
    min_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    
    # Training loop
    history = {'loss': [], 'val_loss': []}
    
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
            best_model_state = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
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

def main():
    # Load data
    dataframe = pd.read_csv("information.csv", delimiter=",", header=None)
    dataset = dataframe.values
    
    x_data = dataset[:, :-1]
    y_data = dataset[:, -1]
    
    # Split data
    train_ratio = 0.70  
    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1 - train_ratio)
    
    print(f"Training data shape: {X_train.shape}")
    
    # Default batch size for initial GA evaluation
    batch_size = 32
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and run the genetic algorithm
    input_size = x_data.shape[1]
    print("Starting genetic algorithm optimization...")
    
    ga = GeneticAlgorithm(
        input_size=input_size,
        train_loader=train_loader,
        val_loader=val_loader,
        population_size=15,  # Smaller for faster execution, increase for better results
        generations=7,       # Number of generations to evolve
        mutation_rate=0.2,   # Probability of mutation
        elite_size=2         # Number of best architectures to keep unchanged
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
    
    # Save model
    save_model = input("Do you want to save the model? (y/n): ")
    if save_model.lower() == 'y':
        torch.save(best_model.state_dict(), 'neural_network_ga_optimized.pth')
        
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
