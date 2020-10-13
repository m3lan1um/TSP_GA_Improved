#!/usr/bin/env python
# coding: utf-8

# <hr style="border:2px solid gray"> </hr>
#
# # Homework 1 - Traveling Salesman Problem
#
# ## Example Code
#
# ### Algorithm 4: Genetic Algorithm
#
# ### Author: Wangduk Seo (CAU AI Lab)
# <hr style="border:2px solid gray"> </hr>

# # Step 0. Importing packages and Global Settings

# In[ ]:


# package list
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sys
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import time

# Global Variables
# Genetic Algorithm
POOL_SIZE = 100
RANDOM_RATIO = 0.1
ITERATIONS = 20000
K = 10
np.random.seed(0)

# Plot Settings
PLOT_MODE = False # Draw Route
PLT_INTERVAL = 100 # Draw Route every 100 iterations
plt.ion()

# First City Index
FIRST_IDX = 0


# # Step 1. Data Loading

# In[ ]:


def fileloader():
    # Data loading
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    if file_path == '':
        raise Exception('Cannot load a data file')
    root.destroy()
    #     Data Format
    #     ---------------------------------------------------------
    #     NAME : pia3056
    #     COMMENT : Bonn VLSI data set with 3056 points
    #     COMMENT : Uni Bonn, Research Institute for Discrete Math
    #     COMMENT : Contributed by Andre Rohe
    #     TYPE : TSP
    #     DIMENSION : 3056 -----------------------------|
    #     EDGE_WEIGHT_TYPE : EUC_2D                     |
    #     NODE_COORD_SECTION                            |
    #     1 0 11 (2 dimentional coordinate of city)     |
    #     2 0 115                                       |
    #     ...                                           |
    #     ...(Total 3056 nodes)<------------------------|
    #     EOF
    #     ---------------------------------------------------------
    with open(file_path, "r") as file:
        file_str = file.readlines()

    # Get the coordinates of cities
    coord_str = file_str[8:-1]  # first city string to last city string (EOF 전까지)
    coord_list = np.zeros((len(coord_str), 2))
    for idx, item in enumerate(coord_str):
        coord_split = item.split()
        coord_list[idx, 0] = int(coord_split[1])
        coord_list[idx, 1] = int(coord_split[2])

    return coord_list


# # Step 2. Initialization

# In[ ]:


def path_cost(path_map, path):
    # The array of cost between cities in the path
    cnt_cities = path_map.shape[0]
    cost_arr = np.zeros(cnt_cities)
    for i in range(cnt_cities):
        cost_arr[i] = path_map[path[i], path[i+1]]

    return cost_arr


def initialize_greedy(coord_list, first_idx):
    cnt_cities = len(coord_list)

    # Initialize path and insert first city index to the first and last elements
    path_map = euclidean_distances(coord_list, coord_list)

    ave_list = np.zeros(cnt_cities, dtype=np.float)
    std_list = np.zeros(cnt_cities, dtype=np.float)
    for idx in range(cnt_cities):
        ave_list[idx] = np.mean(path_map[idx])
        std_list[idx] = np.std(path_map[idx])
    sorted_ave_list = np.argsort(ave_list)
    sorted_std_list = np.argsort(std_list)

    rank_list = np.zeros(cnt_cities, dtype=np.int)
    for rank in range(cnt_cities):
        rank_list[sorted_ave_list[rank]] += rank
        rank_list[sorted_std_list[rank]] += rank
    sorted_priority_list = np.argsort(rank_list)
    first_city_idx = sorted_priority_list[-first_idx]

    path = np.zeros(cnt_cities + 1, dtype=np.int)
    path[0], path[-1] = first_city_idx, first_city_idx

    # Euclidean distance map between cities

    cities_tovisit = np.ones((cnt_cities), dtype=np.bool)
    cities_tovisit[first_city_idx] = False

    # Iteratively Connect nearest cities
    for i in range(1, cnt_cities):
        start_idx = path[i - 1]
        distance_from_start = path_map[start_idx, :]
        nearest_list = np.argsort(distance_from_start)
        for idx in range(len(nearest_list)):
            # check the nearest city is visited
            if cities_tovisit[nearest_list[idx]]:
                nearest_city = nearest_list[idx]
                break
        cities_tovisit[nearest_city] = False
        path[i] = nearest_city

    return path_map, path


def initialize_random(coord_list, first_idx):
    cnt_cities = len(coord_list)
    path = np.zeros(cnt_cities + 1, dtype=np.int)

    path[0], path[-1] = first_idx, first_idx
    # Euclidean distance map between cities
    path_map = euclidean_distances(coord_list, coord_list)

    # city indices without first city index
    cities_tovisit = np.delete(np.arange(cnt_cities), first_idx)
    cities_random = np.random.permutation(cities_tovisit)
    path[1:-1] = cities_random

    return path_map, path


def two_opt_swap(path_map, path, steps):
    cnt_cities = path_map.shape[0]
    curr_path = path.copy()
    for i in range(steps):
        # Select two indices for flip points
        sel_idx = np.sort(np.random.choice(np.arange(1, cnt_cities + 1), 2))
        if sel_idx[1] - sel_idx[0] <= 1:
            continue
        # Path Flip and update cost array
        curr_path[sel_idx[0]:sel_idx[1]] = np.flip(curr_path[sel_idx[0]: sel_idx[1]])

    cost_arr = path_cost(path_map, curr_path)
    curr_cost = cost_arr.sum()

    return curr_path, curr_cost


def initialization(coord_list):
    # Greedy Search + two_opt + random
    cnt_cities = len(coord_list)
    path_pool = np.zeros((POOL_SIZE, cnt_cities + 1), dtype=np.int)
    pool_cost = np.zeros(POOL_SIZE)
    num_random = int(POOL_SIZE * RANDOM_RATIO)

    path_map, path_pool[0, :] = initialize_greedy(coord_list, FIRST_IDX)
    pool_cost[0] = path_cost(path_map, path_pool[0, :]).sum()
    for i in range(1, num_random + 1):
        _, path_pool[i, :] = initialize_greedy(coord_list, i + 1)
        pool_cost[i] = path_cost(path_map, path_pool[i, :]).sum()

    for i in range(num_random + 1, POOL_SIZE):
        path_pool[i, :], pool_cost[i] = two_opt_swap(path_map, path_pool[0, :], np.random.randint(10, 25))

    return path_map, path_pool, pool_cost


# # Step 3. Selection
# ## Tournament Selection

# In[ ]:


def selection(pool_cost, k):
    # tournament selection
    selected_indices = np.random.permutation(POOL_SIZE)
    selected_indices = selected_indices[:k]

    selected_cost = pool_cost[selected_indices]
    sorted_indices = np.argsort(selected_cost)

    indices = selected_indices[sorted_indices[:2]]

    return indices


# # Step 4. Crossover
# ## PMX Crossover

# In[ ]:


def crossover(path1, path2):
    # pmx crossover
    path_size = len(path1) - 1
    plist1 = np.zeros(path_size, dtype=np.int)
    plist2 = np.zeros(path_size, dtype=np.int)

    child1 = path1.copy()
    child2 = path2.copy()

    for i in range(path_size):
        plist1[child1[i]] = i
        plist2[child2[i]] = i

    sel_idx = np.random.randint(1, path_size, size=2)
    if sel_idx[0] > sel_idx[1]:
        sel_idx[0], sel_idx[1] = sel_idx[1], sel_idx[0]

    for i in range(sel_idx[0], sel_idx[1]):
        # Swap Points
        temp1, temp2 = child1[i], child2[i]
        child1[i], child1[plist1[temp2]] = temp2, temp1
        child2[i], child2[plist2[temp1]] = temp1, temp2
        plist1[temp1], plist1[temp2] = plist1[temp2], plist1[temp1]
        plist2[temp1], plist2[temp2] = plist2[temp2], plist2[temp1]

    return child1, child2


# # Step 5. Mutation
# ## Swap Mutation

# In[ ]:


def mutation(path):
    # Swap mutation
    path_size = len(path) - 1
    sel_idx = np.random.randint(1, path_size, size=2)
    child = path.copy()
    child[sel_idx[0]], child[sel_idx[1]] = child[sel_idx[1]], child[sel_idx[0]]

    return child


# # Step 6. Searching a path
#
# ## Algorithm 4. GA

# In[ ]:


def ga_search(coord_list):
    # Initialization
    path_map, path_pool, pool_cost = initialization(coord_list)

    for i in range(ITERATIONS):
        sort_cost = np.argsort(pool_cost)
        path_pool = path_pool[sort_cost, :]
        pool_cost = pool_cost[sort_cost]

        # Selection
        indices = selection(pool_cost, K)
        # Crossover
        child1, child2 = crossover(path_pool[indices[0], :], path_pool[indices[1], :])
        # Mutation
        child3 = mutation(path_pool[indices[0], :])

        path_pool[-1, :], pool_cost[-1] = child1, path_cost(path_map, child1).sum()
        path_pool[-2, :], pool_cost[-2] = child2, path_cost(path_map, child2).sum()
        path_pool[-3, :], pool_cost[-3] = child3, path_cost(path_map, child3).sum()

    sort_cost = np.argsort(pool_cost)
    path_pool = path_pool[sort_cost, :]
    pool_cost = pool_cost[sort_cost]

    return path_pool[0], pool_cost[0]


# # Main

# In[ ]:


try:
    coord_list = fileloader()
except Exception as e:
    print('예외 발생', e)
    sys.exit()

start_time = time.time()

best_path, best_cost = ga_search(coord_list)

if PLOT_MODE:
    figure, ax = plt.subplots()
    plt.scatter(coord_list[:, 0], coord_list[:, 1], c='yellow', s=10)
    plt.title('City Route')
    coord_path = coord_list
    coord_path = np.append(coord_path, coord_path[best_path[0], :].reshape(1, 2), axis=0)
    coord_path[:, :] = coord_path[best_path, :]
    lines, = ax.plot(coord_path[:, 0], coord_path[:, 1], 'k--')
    figure.canvas.draw()

print('Execution Time: ' + str(time.time() - start_time))
print('Path: ' + str(best_path.tolist()))
print('Cost: ' + str(best_cost))


# In[ ]:




