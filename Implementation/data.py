import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Leemos el dataset de precios de cierre de acciones
df = pd.read_csv('C:/Users/ASUS/OneDrive - Universidad EAFIT/2023-1/OPTI II/Proyecto/Multi-objective-optimization-for-portfolio-selection/Data/all_stocks_5yr.csv')
print(df)
df = df[['date', 'Name', 'close']]
df_pivot = df.pivot(index='date', columns='Name', values='close')

def calculate_return(weights, returns):
    """
    Calcula la rentabilidad de un portafolio a partir de los precios de cierre y los pesos de cada acción.
    """
    print(np.dot(weights, returns).shape)
    return np.dot(weights, returns)

def calculate_risk(weights, cov_matrix):
    """
    Calcula el riesgo de un portafolio a partir de los precios de cierre y los pesos de cada acción.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Definimos los parámetros de NSGA-II
pop_size = 505
n_generations = 50
crossover_rate = 0.9
mutation_rate = 1.0 / pop_size
# Generamos la población inicial de portafolios aleatorios

#n_assets = len(df_pivot.columns)
n_assets=len(df_pivot.columns)
population = np.zeros((pop_size, n_assets))
# Generar una población inicial de portafolios aleatorios
population = np.random.rand(pop_size, n_assets)
population = population / np.sum(population, axis=1)[:, None]
population = population
print(population.shape)

# Calculamos los retornos y covarianzas de las acciones
returns = df_pivot.pct_change().dropna()
returns=returns.T
cov_matrix = df_pivot.pct_change().cov()
print(returns.shape)

# Evaluamos la función objetivo y las restricciones para cada portafolio en la población inicial
fitness = np.zeros((pop_size, 2))
for i, portfolio in enumerate(population):
    fitness[i, 0] = -1 * calculate_return(portfolio, returns) # Maximizar la rentabilidad
    fitness[i, 1] = calculate_risk(portfolio, cov_matrix) # Minimizar el riesgo
print(fitness)