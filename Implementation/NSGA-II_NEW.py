import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Leemos el dataset de precios de cierre de acciones
df = pd.read_csv('C:/Users/ASUS/OneDrive - Universidad EAFIT/2023-1/OPTI II/Proyecto/Multi-objective-optimization-for-portfolio-selection/Data/all_stocks_5yr.csv', index_col='date')
print(df)

def calculate_return(weights, returns):
    """
    Calcula la rentabilidad de un portafolio a partir de los precios de cierre y los pesos de cada acción.
    """
    return np.dot(weights, returns)

def calculate_risk(weights, cov_matrix):
    """
    Calcula el riesgo de un portafolio a partir de los precios de cierre y los pesos de cada acción.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Definimos los parámetros de NSGA-II
pop_size = 100
n_generations = 50
crossover_rate = 0.9
mutation_rate = 1.0 / pop_size

# Definir la función de selección NSGA-II
def nsga2_selection(population, fitness, pop_size, n_offspring):
    # Calcular la dominancia de cada individuo en la población
    dominance = np.zeros(pop_size, dtype=int)
    for i in range(pop_size):
        for j in range(pop_size):
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominance[i] += 1
            elif np.all(fitness[i] >= fitness[j]) and np.any(fitness[i] > fitness[j]):
                dominance[i] -= 1
    
    # Seleccionar los individuos no dominados de la población
    fronts = []
    current_front = []
    for i in range(pop_size):
        if dominance[i] == 0:
            current_front.append(i)
    
    while current_front:
        next_front = []
        for i in current_front:
            for j in range(pop_size):
                if i == j:
                    continue
                if dominance[j] > 0:
                    dominance[j] -= 1
                if dominance[j] == 0:
                    next_front.append(j)
                    dominance[j] = -1
        fronts.append(current_front)
        current_front = next_front
    
    # Seleccionar los mejores individuos de los frentes no dominados
    selected = []
    remaining = n_offspring
    for front in fronts:
        if remaining < len(front):
            selected.extend(random.sample(front, remaining))
            break
        else:
            selected.extend(front)
            remaining -= len(front)
    
    return selected

def nsga2_crossover(parent_1, parent_2, crossover_rate):
    child = np.zeros_like(parent_1)
    for i in range(len(parent_1)):
        if random.random() < crossover_rate:
            child[i] = parent_1[i]
        else:
            child[i] = parent_2[i]
    return child

def nsga2_mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = np.clip(individual[i] + np.random.normal(scale=0.1), 0, 1)
    return individual

# Generamos la población inicial de portafolios aleatorios
n_assets = len(df.columns)
population = [np.random.rand(n_assets) for i in range(pop_size)]
population = [x / sum(x) for x in population]

# Calculamos los retornos y covarianzas de las acciones
returns = df.pct_change().mean()
cov_matrix = df.pct_change().cov()

# Evaluamos la función objetivo y las restricciones para cada portafolio en la población inicial
fitness = np.zeros((pop_size, 2))
for i, portfolio in enumerate(population):
    fitness[i, 0] = -1 * calculate_return(portfolio, returns) # Maximizar la rentabilidad
    fitness[i, 1] = calculate_risk(portfolio, cov_matrix) # Minimizar el riesgo

# Ejecutamos el algoritmo NSGA-II
for gen in range(n_generations):
    offspring = []
    for i in range(pop_size):
        # Seleccionamos los mejores portafolios mediante la función de selección NSGA-II
        selected_parents = nsga2_selection(population, fitness, pop_size, 2)
        parent_1 = population[selected_parents[0]]
        parent_2 = population[selected_parents[1]]

        # Realizar la cruza y la mutación de los portafolios seleccionados para crear la siguiente generación
        child = nsga2_crossover(parent_1, parent_2, crossover_rate)
        child = nsga2_mutation(child, mutation_rate)
        child /= child.sum()
        offspring.append(child)
        
    # Evaluar la función objetivo y las restricciones de cada portafolio en la población de hijos
    offspring_fitness = np.zeros((pop_size, 2))
    for i, portfolio in enumerate(offspring):
        offspring_fitness[i, 0] = -1 * calculate_return(portfolio, returns) # Maximizar la rentabilidad
        offspring_fitness[i, 1] = calculate_risk(portfolio, cov_matrix) # Minimizar el riesgo


# Unir la población actual y la población de hijos
combined_population = np.concatenate((population, offspring))
combined_fitness = np.concatenate((fitness, offspring_fitness))

# Realizar la selección de los mejores portafolios para la siguiente generación mediante la función de selección NSGA-II
population, fitness = nsga2_selection(combined_population, combined_fitness, pop_size, 2, True)

best_portfolio = population[0]
print("Rentabilidad del mejor portafolio: {:.4f}".format(calculate_return(best_portfolio)))
print("Riesgo del mejor portafolio: {:.4f}".format(calculate_risk(best_portfolio)))
for i, weight in enumerate(best_portfolio):
    print("Ponderación del activo {}: {:.4f}".format(n_assets[i], weight))
    
plt.scatter(fitness[:, 0], fitness[:, 1], alpha=0.5)
plt.scatter(-fitness[:, 0], fitness[:, 1], alpha=0.5)
plt.scatter(-calculate_return(best_portfolio, returns), calculate_risk(best_portfolio, cov_matrix), color='red', s=100)
plt.xlabel('Retorno')
plt.ylabel('Riesgo')
plt.show()