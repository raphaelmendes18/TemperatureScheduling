
# -*- coding: utf-8 -*-
'''
Define units
---------------------------------------------------------------------------------------------------------------------
Vmax = 5;            % The highest temperature we can reach                       [°C]
Vhigh = 4;           % The highest temperature we can reach                       [°C]
Vlow = 0;            % The highest temperature we can reach                       [°C]
Vmin = -1;           % The lowest temperature we can reach                        [°C]
V0 = 0;              % Inetial temperature at time = 0
K = 2;               % Number of heaters 
Tmax = 7;            % Total time                                                 [h]
PiP = [50 50];       % Penalty cost                                               [$/h]
PiC = [10 20];       % Continuous cost                                            [$/h]
PiD = [30 10];       % Discrete cost                                              [$]
A = [4/3 2];         % Heating parameters for mode 1,2                            [°C/h]
A0 = -4;             % Cooling parameters for mode 0, when the heaters are off    [°C/h]
---------------------------------------------------------------------------------------------------------------------
'''
import numpy as np
import time
from ga_functions import *
import dask

Vmax, Vhigh, Vlow, Vmin = 5, 4, 0, -1 # Temperature parameters
V0 = 0 # Initial temperature
K = 2 # Number of heaters
Tmax = 7 # Total time

PiP = np.array([50.0, 50.0]) # Penalty Cost for each Temperature danger zone
PiC = np.array([0.0, 10.0, 20.0]) # Continuous cost for each mode
PiD = np.array([0.0, 30.0, 10.0]) # Discrete cost for each mode when changing to this mode
A = np.array([-4.0, 4.0/3.0, 2.0]) # Heating parameters

# Genetic Algorithm Parameters
pop_size = 1000
mutation_rate = 0.2
crossover_rate = 0.6
tour = 0.2
generations = 10
runs = 50 # If you want to run the entire genetic algorith many times, usually this ensures that you have good solutions and that the GA is stable

# Genetic algorithm run in parallel
results = []
tic = time.time()
try: 
    for r in range(runs):
        parallel_runs = dask.delayed(genetic_algorithm)(pop_size, crossover_rate, mutation_rate, generations, tour, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax, K, A)
        results.append(parallel_runs)
    ga_results = dask.compute(*results, scheduler='processes', num_workers=8)
except AttributeError:
    print('Warning - DASK not available, running single threaded.')
    ga_results = []
    for r in range(runs):
        ga_results.append(genetic_algorithm(pop_size, crossover_rate, mutation_rate, generations, tour, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax, K, A))
finally:
    toc = time.time()    
    print_results(ga_results, Vmax, Vhigh, Vlow, Vmin, V0, K, Tmax, PiP, PiC, PiD, A, runs, pop_size, mutation_rate, crossover_rate, tour, generations)
    print('Total Run Time {} (s)'.format(toc-tic))