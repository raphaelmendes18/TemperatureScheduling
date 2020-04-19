import numpy as np
import random

def init_population(pop_size, crossover_rate, Tmax, K):
    
    # Valid Modes
    choices = range(0, K+1)

    # Initialize pop_size + crossovers
    # The pop_size*crossover are going to be zero in the initialization to 
    # reduce cost of this function, thus generating only pop_size random solutions

    pop = np.zeros((int(pop_size*(1+crossover_rate)), Tmax))
    for i in range(pop_size):
        for j in range(Tmax):
            if j == 0:
                pop[i,j] = int(random.choice(choices))
            else:
                pop[i,j] = int(random.choice([c for c in choices if c != pop[i,j-1]]))
    return pop

def crossover(pop, pop_size, crossover_rate, tour):

    # Generate offspring 
    for i in range(pop_size, int(pop_size*(1+crossover_rate)), 2):        
        # Pick parents from mating pool of tour best solutions
        parent_1 = random.choice(pop[0:int(pop_size*tour)])
        # print(parent_1)
        parent_2 = random.choice(pop[0:int(pop_size*tour)])
        
        # randomize crossover point
        cp = random.randint(1, pop[i].shape[0]-1)
        pop[i] = np.concatenate([parent_1[0:cp], parent_2[cp:pop[i].shape[0]]])
        pop[i+1] = np.concatenate([parent_2[0:cp], parent_1[cp:pop[i].shape[0]]])
        # print(pop[i])
    return pop
def mutation(pop, pop_size, mutation_rate, crossover_rate):

    # Mutation is perfomed by performing permutations
    for _ in range(int(pop_size*mutation_rate)):
        el = random.randint(0,int(pop_size*(1+crossover_rate)-1))
        pop[el] = np.random.permutation(pop[el])

    return pop
def calculate_temperatures(pop, pop_size, crossover_rate, A, Tmax):

    # Trasform the sequence of modes in temperature to calculate
    # cost later
    V = np.zeros((int(pop_size*(1+crossover_rate)), Tmax+1))
    for i in range(V.shape[0]):
        for j in range(1, V.shape[1]):
            V[i,j] = V[i,j-1] + A[int(pop[i,j-1])]

    return V

def fitness(pop, V, pop_size, crossover_rate, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax):
    # Calculate the fitness using the cost function
    fitness_arr = np.zeros(int(pop_size*(1+crossover_rate)))

    for idx, _ in enumerate(fitness_arr):
        fitness_arr[idx] = cost(pop[idx], V[idx], Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax)

    return fitness_arr
    

def cost(alpha, V, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax):

    cost = np.zeros(Tmax)
    cost = np.add(list(map(lambda f: PiD[int(f)], alpha)), list(map(lambda f: PiC[int(f)], alpha)))
    
    for t in range(1,Tmax):
        if alpha[t] == alpha[t-1]:
            cost_invalid_solution = np.inf
            break
        else:
            cost_invalid_solution = 0
    if V[Tmax] > 0.1: # approximating due to 1.33 in gas mode
        cost_invalid_solution = np.inf

    return cost_invalid_solution + np.sum(cost) + penalty_cost(V, Vmax, Vhigh, Vlow, Vmin, PiP, Tmax)

def penalty_cost(V, Vmax, Vhigh, Vlow, Vmin, PiP, Tmax):

    # Get penalizing temperature thresholds
    def p_cost(v, Vhigh, Vlow, PiP):
        if Vmax < v:
            return np.inf
        elif Vmin > v:
            return np.inf
        elif Vhigh < v:
            return PiP[1]
        elif Vlow > v:
            return PiP[0]
        else:
            return 0
    
    return np.sum(list(map(lambda v: p_cost(v, Vhigh, Vlow, PiP), V)))

def natural_selection(pop, V, fitness_arr):
    # Sort arrays based on fitness
    fit_arr_indexes = np.argsort(fitness_arr)

    # Create new arrays to store copies sorted
    new_pop = np.zeros(len(pop))
    new_V = np.zeros(len(V))
    new_fitness = np.zeros(len(fitness_arr))

    new_pop = np.array(list(map(lambda i: pop[i], fit_arr_indexes)))
    new_V = np.array(list(map(lambda i: V[i], fit_arr_indexes)))
    new_fitness = np.array(list(map(lambda i: fitness_arr[i], fit_arr_indexes)))

    return new_pop, new_V, new_fitness

def print_results(results, Vmax, Vhigh, Vlow, Vmin, V0, K, TMax, PiP, PiC, PiD, A, runs, pop_size, mutation_rate, crossover_rate, tour, generations):

    print('####################################')
    print('######---Genetic Algorithm---#######')
    print('######-----------------------#######')
    print('######-Scheduling Parameters-#######')
    print('##Vlow: {}                       '.format(Vlow))
    print('##Vmin: {}                       '.format(Vmin))
    print('##Vmax: {}                       '.format(Vmax))
    print('##Vhigh: {}                      '.format(Vhigh))
    print('##V0: {}                     '.format(V0))
    print('##K: {}                      '.format(K))
    print('##PiP: {}                        '.format(PiP))
    print('##PiC: {}                        '.format(PiC))
    print('##PiD: {}                        '.format(PiD))
    print('##A: {}                      '.format(A))
    print('######-Genetic Algorithm Parameters-#######')
    print('##Total Runs: {}'.format(runs))
    print('##Pop Size: {}'.format(pop_size))
    print('##Mutation Rate: {}'.format(mutation_rate))
    print('##Crossover Rate: {}'.format(crossover_rate))
    print('##Tour: {}'.format(tour))
    print('##Generations: {}'.format(generations))
    for i, r in enumerate(results):
        print('##############Run {} ###############'.format(i))
        print('##Schedule: {} '.format("->".join(["{}".format(m) for m in r[1]])))
        print('##Temperatures: {} '.format("->".join("{:.2f}".format(t) for t in r[2])))
        print('##Cost: ${:.2f} USD '.format(r[0]))

def genetic_algorithm(pop_size, crossover_rate, mutation_rate, generations, tour, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax, K, A):       
    pop = init_population(pop_size, crossover_rate, Tmax, K)
    for i in range(generations):
        pop = crossover(pop, pop_size, crossover_rate, tour)
        pop = mutation(pop, pop_size, mutation_rate, crossover_rate)
        V = calculate_temperatures(pop, pop_size, crossover_rate, A, Tmax)
        fit_arr = fitness(pop, V, pop_size, crossover_rate, Vmax, Vhigh, Vlow, Vmin, PiP, PiC, PiD, Tmax)
        pop, V, fit_arr = natural_selection(pop, V, fit_arr)

    return fit_arr[0], pop[0], V[0]
