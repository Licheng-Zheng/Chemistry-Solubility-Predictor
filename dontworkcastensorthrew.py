import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate dummy input data
def generate_input(batch_size=32):
    return torch.randn(batch_size, 120)

# Define the evaluation function
def evaluate(individual):
    # Create model and convert individual to tensor format
    model = Net()
    params = list(model.parameters())
    
    # Convert individual to proper shape and type
    individual_t = torch.FloatTensor(individual)
    
    # Update model parameters
    index = 0
    for param in params:
        num_elements = param.numel()
        param_data = individual_t[index:index + num_elements].view(param.size())
        param.data.copy_(param_data)
        index += num_elements
    
    # Training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Generate training data
    inputs = generate_input()
    targets = torch.randn(inputs.size(0), 1)  # Random targets for demonstration
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Return loss as fitness value (to be minimized)
    return (loss.item(),)

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Initialize population with weights in [-1, 1] range
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=sum(p.numel() for p in Net().parameters()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Genetic Algorithm execution
def main():
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    for gen in range(ngen):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        
        # Get all fitness values
        fitness_values = [ind.fitness.values[0] for ind in population]
        min_fit = min(fitness_values)
        max_fit = max(fitness_values)
        avg_fit = np.mean(fitness_values)
        med_fit = np.median(fitness_values)
        
        print(f"  Fitness: Min={min_fit:.2f}, Max={max_fit:.2f}, Avg={avg_fit:.2f}, Med={med_fit:.2f}")

    best_ind = tools.selBest(population, k=1)[0]
    print("Best individual is: %s\nwith fitness: %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()