import numpy as np
import random

# Definir los parámetros de NSGA-II
pop_size = 100
n_generations = 50
crossover_rate = 0.9
mutation_rate = 1.0 / pop_size

# Generar una población inicial aleatoria de portafolios
def generate_random_portfolio(n_assets):
    portfolio = np.random.rand(n_assets)
    portfolio /= portfolio.sum()
    return portfolio

population = [generate_random_portfolio(n_assets) for i in range(pop_size)]

# Evaluar la función objetivo y las restricciones de cada portafolio en la población inicial
fitness = np.zeros((pop_size, 2))
for i, portfolio in enumerate(population):
    fitness[i, 0] = -1 * calculate_return(portfolio) # Maximizar la rentabilidad
    fitness[i, 1] = calculate_risk(portfolio) # Minimizar el riesgo

# Realizar el algoritmo NSGA-II para generar la siguiente generación de portafolios
for gen in range(n_generations):
    offspring = []
    for i in range(pop_size):
        # Seleccionar los mejores portafolios mediante la función de selección NSGA-II
        selected_parents = nsga2_selection(population, fitness, pop_size, 2)
        parent_1 = population[selected_parents[0]]
        parent_2 = population[selected_parents[1]]
        
        # Realizar la cruza y la mutación de los portafolios seleccionados para crear la siguiente generación
        child = nsga2_crossover(parent_1, parent_2, crossover_rate)
        child = nsga2_mutation(child, mutation_rate)
        child /= child.sum()
        offspring.append(child)
        
    # Evaluar la función objetivo y
