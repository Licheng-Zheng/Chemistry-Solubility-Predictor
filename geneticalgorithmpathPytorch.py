import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import h5py

class Individual:
    def __init__(self, genes):
        self.genes = genes  # Parameters to optimize
        self.fitness = 0.0  # Fitness score

class GeneticAlgorithmOptimizer:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.history = {'fitness': []}  # Initialize history tracking
        
    def initialize_population(self, gene_length):
        self.population = [Individual(np.random.uniform(-1, 1, gene_length)) for _ in range(self.population_size)]
        
    def evaluate_fitness(self, individual, dataset):
        # Calculate fitness based on dataset performance
        # Here we'll use mean squared error as fitness metric
        predictions = dataset.x * individual.genes[0] + individual.genes[1]
        mse = mean_squared_error(dataset.y, predictions)
        individual.fitness = 1 / (mse + 1e-8)  # Higher is better
        
    def select_parents(self):
        # Tournament selection
        parents = []
        for _ in range(self.population_size // 2):
            tournament = np.random.choice(self.population, 5)
            parents.append(max(tournament, key=lambda x: x.fitness))
        return parents
        
    def crossover(self, parent1, parent2):
        # Simple arithmetic crossover
        child_genes = parent1.genes * (1 - self.crossover_rate) + parent2.genes * self.crossover_rate
        return Individual(child_genes)
        
    def mutate(self, individual):
        # Gaussian mutation
        mutation_mask = np.random.binomial(1, self.mutation_rate, individual.genes.shape)
        individual.genes += mutation_mask * np.random.normal(0, 0.1, individual.genes.shape)
        
    def evolve(self, dataset, generations=50):
        gene_length = 2  # Number of parameters to optimize
        self.initialize_population(gene_length)
        
        for generation in range(generations):
            # Evaluate fitness
            for individual in self.population:
                self.evaluate_fitness(individual, dataset)
                
            # Record best fitness
            best_individual = max(self.population, key=lambda x: x.fitness)
            self.history['fitness'].append(best_individual.fitness)
                
            # Select parents and create next generation
            parents = self.select_parents()
            next_generation = []
            
            for i in range(0, len(parents)-1, 2):
                parent1 = parents[i]
                parent2 = parents[i+1]
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)
                
            self.population = next_generation + parents[:self.population_size - len(next_generation)]
            
            # Print progress
            print(f"Generation {generation+1}: Best fitness = {best_individual.fitness:.4f}")
            
        return best_individual

    def save_to_hdf5(self, filename):
        """Save the optimizer state to an HDF5 file"""
        with h5py.File(filename, 'w') as f:
            # Save optimizer parameters
            f.create_dataset('population_size', data=self.population_size)
            f.create_dataset('mutation_rate', data=self.mutation_rate)
            f.create_dataset('crossover_rate', data=self.crossover_rate)
            
            # Save history
            history_group = f.create_group('history')
            for key, value in self.history.items():
                history_group.create_dataset(key, data=value)
                
        print(f"Optimizer parameters saved to {filename}")

    @staticmethod
    def load_from_hdf5(filename):
        """Load optimizer parameters from an HDF5 file"""
        with h5py.File(filename, 'r') as f:
            population_size = f['population_size'][()]
            mutation_rate = f['mutation_rate'][()]
            crossover_rate = f['crossover_rate'][()]
            
            # Load history
            history = {}
            for key in f['history'].keys():
                history[key] = f['history'][key][()]
        
        optimizer = GeneticAlgorithmOptimizer(
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        optimizer.history = history
        
        return optimizer

class RegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

class OptimizedRegressionDataset(RegressionDataset):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.scaling_factor = 1.0
        self.noise_level = 0.0
        self.history = {}  # Initialize history
        self.optimize()
        
    def optimize(self):
        # Initialize genetic algorithm
        ga = GeneticAlgorithmOptimizer(population_size=50, mutation_rate=0.1)
        
        # Evolve dataset parameters
        best_individual = ga.evolve(self, generations=50)
        
        # Update dataset parameters with best values
        self.scaling_factor = best_individual.genes[0]
        self.noise_level = best_individual.genes[1]
        
        # Store optimization history
        self.history = ga.history
        
    def __getitem__(self, idx):
        x_sample = self.x[idx] * self.scaling_factor + np.random.normal(0, self.noise_level)
        y_sample = self.y[idx]
        return x_sample, y_sample

def main():
    # Example usage
    x_data = np.linspace(0, 10, 100).reshape(-1, 1)
    y_data = 2 * x_data + 1 + np.random.normal(0, 1, x_data.shape)
    
    dataset = OptimizedRegressionDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Save the optimizer
    ga = GeneticAlgorithmOptimizer()
    ga.save_to_hdf5('optimizer.h5')
    
    # Visualize optimization progress
    plt.figure(figsize=(10, 6))
    plt.plot(dataset.history['fitness'], label='Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()