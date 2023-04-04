import pandas as pd
import numpy as np
import random

# Cargar el dataset
df = pd.read_csv("stocks.csv")

# Convertir la columna 'date' a un objeto datetime
df['date'] = pd.to_datetime(df['date'])

# Ordenar el dataset por fecha y por nombre de acción
df = df.sort_values(['date', 'Name'])

# Crear una lista con los nombres de las acciones únicas
stocks = df['Name'].unique()

# Crear un dataframe con los retornos diarios de cada acción
returns = pd.DataFrame(columns=stocks)
for stock in stocks:
    s = df.loc[df['Name'] == stock, 'close']
    returns[stock] = np.log(s) - np.log(s.shift(1))

# Calcular la matriz de covarianzas de los retornos diarios
cov_matrix = returns.cov()

# Definir la función para calcular la rentabilidad de un portafolio
def calculate_return(portfolio, returns):
    return np.dot(portfolio, returns.mean())

# Definir la función para calcular el riesgo de un portafolio
def calculate_risk(portfolio, cov_matrix):
    return np.sqrt(np.dot(portfolio.T, np.dot(cov_matrix, portfolio)))

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

# Definir la función de cruza NSGA
pop_size = 100
n_generations = 50
crossover_rate = 0.9
mutation_rate = 1.0 / pop_size

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

    # Evaluar la función objetivo y las restricciones de cada portafolio en la población siguiente
    fitness_offspring = np.zeros((pop_size, 2))
    for i, portfolio in enumerate(offspring):
        fitness_offspring[i, 0] = -1 * calculate_return(portfolio) # Maximizar la rentabilidad
        fitness_offspring[i, 1] = calculate_risk(portfolio) # Minimizar el riesgo

    # Combinar la población actual y la población siguiente
    combined_population = np.concatenate((population, offspring), axis=0)
    combined_fitness = np.concatenate((fitness, fitness_offspring), axis=0)

    # Realizar la selección y el reemplazo de los portafolios para generar la siguiente población
    population, fitness = nsga2_replace(combined_population, combined_fitness, pop_size)

# Seleccionar el portafolio más eficiente de la última población
best_portfolio = population[np.argmax(fitness[:, 0])]

return best_portfolio, -1 * fitness[np.argmax(fitness[:, 0]), 0], fitness[np.argmax(fitness[:, 0]), 1]

