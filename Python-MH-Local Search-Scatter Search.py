############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-Scatter Search

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-Scatter_Search, File: Python-MH-Local Search-Scatter Search.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-Scatter_Search>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
import os
from matplotlib import pyplot as plt 
from operator import itemgetter

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = pd.DataFrame(np.zeros((coordinates.shape[0], coordinates.shape[0])))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates.iloc[i,:]
                y = coordinates.iloc[j,:]
                Xdata.iloc[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = Xdata.copy(deep = True)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m.iloc[i,j] = (1/2)*(Xdata.iloc[0,j]**2 + Xdata.iloc[i,0]**2 - Xdata.iloc[i,j]**2)    
    m = m.values
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    coordinates = coordinates.values
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Selection
#def roulette_wheel_tsp(reference_list):
    #fitness = pd.DataFrame(np.zeros((len(reference_list), 2)), columns = ['Seed', 'Probability'])
    #for i in range(0, fitness .shape[0]):
        #fitness.iloc[i,0] = i
        #fitness.iloc[i,1] = reference_list[i][1]
    #for i in range(0, fitness.shape[0]):
        #fitness.iloc[i,1] = 1/(1 + fitness.iloc[i,1])
    #fit_sum = fitness.iloc[:,1].sum()
    #for i in range(1, fitness.shape[0]):
        #fitness.iloc[i,1] = (fitness.iloc[i,1] + fitness.iloc[i-1,1]) 
    #for i in range(1, fitness.shape[0]):
        #fitness.iloc[i,1] = fitness.iloc[i,1]/fit_sum       
    #ix = 0
    #iy = 0
    #rand_1 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    #for i in range(0, fitness.shape[0]):
        #if (rand_1 <= fitness.iloc[i, 1]):
          #ix = i
          #iy = i
          #break
    #while (ix == iy):
        #rand_2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        #for i in range(0, fitness.shape[0]):
            #if (rand_2 <= fitness.iloc[i, 1]):
              #iy = i
              #break    
    #return ix, iy

# Function: Crossover
def crossover_tsp(Xdata, reference_list, reverse_prob = 0.5, scramble_prob = 0.3):
    #ix, iy = roulette_wheel_tsp(reference_list)
    ix, iy = random.sample(list(range(0,len(reference_list))), 2)
    parent_1 = reference_list[ix][0]
    parent_1 = parent_1[:-1]
    parent_2 = reference_list[iy][0]
    parent_2 = parent_2[:-1]
    offspring = [0]*len(parent_2)
    child = [[],float("inf")]
    i, j = random.sample(list(range(0,len(parent_1))), 2)
    if (i > j):
        i, j = j, i
    rand_1 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    if (rand_1 < reverse_prob):
        parent_1[i:j+1] = list(reversed(parent_1[i:j+1]))
    offspring[i:j+1] = parent_1[i:j+1]
    parent_2 = [x for x in parent_2 if x not in parent_1[i:j+1]]
    rand_2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    if (rand_2 < scramble_prob):
        random.shuffle(parent_2)
    count = 0
    for i in range(0, len(offspring)):
        if (offspring[i] == 0):
            offspring[i] = parent_2[count]
            count = count + 1
    offspring.append(offspring[0])
    child[0] = offspring
    child[1] = distance_calc(Xdata, child)
    return child

# Function: Mutation 2_opt Stochastic
#def mutation_tsp_2_opt_stochastic(Xdata, city_tour, mutation_prob = 0.1):
    #rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    #city_list = copy.deepcopy(city_tour)
    #if (rand < mutation_prob):
        #best_route = copy.deepcopy(city_list)       
        #m, n  = random.sample(range(0, len(city_tour[0])-1), 2)
        #if (m > n):
            #m, n = n, m
        #best_route[0][m:n+1] = list(reversed(best_route[0][m:n+1]))           
        #best_route[0][-1]  = best_route[0][0]              
        #best_route[1] = distance_calc(Xdata, best_route)                     
        #city_list[1] = copy.deepcopy(best_route[1])
        #for k in range(0, len(city_list[0])): 
            #city_list[0][k] = best_route[0][k]          
    #return city_list

# Function: Local Improvement 2_opt
def local_search_2_opt(Xdata, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]  = best_route[0][0]                          
            best_route[1] = distance_calc(Xdata, best_route)           
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    return city_list

# Function: Scatter Search
def scatter_search(Xdata, city_tour, iterations = 50, reference_size = 25, reverse_prob = 0.5, scramble_prob = 0.3):
    count = 0
    best_solution = copy.deepcopy(city_tour)
    reference_list = []
    for i in range(0, reference_size):
        reference_list.append(seed_function(Xdata))
    
    while (count < iterations):            
        candidate_list = []
        for i in range(0, reference_size):
            candidate_list.append(crossover_tsp(Xdata, reference_list = reference_list, reverse_prob = reverse_prob, scramble_prob = scramble_prob))   
        #for i in range(0, reference_size):
            #candidate_list[i] = mutation_tsp_2_opt_stochastic(Xdata, city_tour = candidate_list[i], mutation_prob = mutation_prob)        
        for i in range(0, reference_size):
            candidate_list[i] = local_search_2_opt(Xdata, city_tour = candidate_list[i])
        for i in range(0, reference_size):        
            reference_list.append(candidate_list[i])
        reference_list = sorted(reference_list, key = itemgetter(1))
        reference_list = reference_list[:reference_size]
        for i in range(0, reference_size):
            if (reference_list[i][1] < best_solution[1]):
                best_solution = copy.deepcopy(reference_list[i]) 
        count = count + 1
        print("Iteration =", count, "-> Distance =", best_solution[1])
    print("Best Solution =", best_solution)
    return best_solution

######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-Scatter Search-Dataset-01.txt', sep = '\t') # 17 cities = 1922.33
seed = seed_function(X)
lsss = scatter_search(X, city_tour = seed, iterations = 50, reference_size = 15, reverse_prob = 0.5, scramble_prob = 0.3)
plot_tour_distance_matrix(X, lsss) # Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses.

Y = pd.read_csv('Python-MH-Local Search-Scatter Search-Dataset-02.txt', sep = '\t') # Berlin 52 = 7544.37
X = buid_distance_matrix(Y)
seed = seed_function(X)
lsss = scatter_search(X, city_tour = seed, iterations = 150, reference_size = 10, reverse_prob = 0.5, scramble_prob = 0.5)
plot_tour_coordinates (Y, lsss) # Red Point = Initial city; Orange Point = Second City
