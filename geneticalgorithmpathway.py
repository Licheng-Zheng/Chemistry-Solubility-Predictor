import pygad
import numpy as np

# Define the neural network architecture
input_layer_size = 3
hidden_layer_size = 4
output_layer_size = 2

# Define the fitness function
def fitness_func(solution, solution_idx):
    # Decode the solution into weights
    input_hidden_weights = solution[:input_layer_size * hidden_layer_size].reshape(input_layer_size, hidden_layer_size)
    hidden_output_weights = solution[input_layer_size * hidden_layer_size:].reshape(hidden_layer_size, output_layer_size)

    # Example input data
    inputs = np.array([[0.1, 0.2, 0.3]])
    
    # Forward pass
    hidden_layer_output = np.dot(inputs, input_hidden_weights)
    hidden_layer_output = np.tanh(hidden_layer_output)  # Activation function
    output = np.dot(hidden_layer_output, hidden_output_weights)
    output = np.tanh(output)  # Activation function

    # Example target output
    target_output = np.array([[0.5, 0.5]])
    
    # Calculate the error
    error = np.mean(np.square(target_output - output))
    
    # Fitness is the inverse of error
    fitness = 1.0 / (error + 1e-6)
    return fitness

# Genetic Algorithm parameters
num_generations = 100
num_parents_mating = 4
sol_per_pop = 8
num_genes = (input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size)

# Initialize the genetic algorithm
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       mutation_percent_genes=10)

# Run the genetic algorithm
ga_instance.run()

# Retrieve the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Fitness of the best solution:", solution_fitness)