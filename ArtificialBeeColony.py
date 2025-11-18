import numpy as np
import random
from dataclasses import dataclass
from EventList import EventList
from Utils import positional_entropy, topological_sort, calculate_unique_individuals
import copy

@dataclass
class InitParams:
    heuristics_rate: float = 0.0
    sampling_rate: int = 25

@dataclass
class Parameters:
    N: int
    limit: int
    max_evaluations: int
    stagnation: int = 1
    mr: float = 0.1
    l: int = -1
    init_params: InitParams = None

class ArtificialBeeColony:
    
    def __init__(self, psmodel):
        self.psmodel = psmodel

    def init_population(self, N, init_params: InitParams):
        proportion = int(init_params.heuristics_rate * N * 0.25)
        sampling = init_params.sampling_rate

        ldf = topological_sort(self.psmodel.jobs, metric='ldf')
        sdf = topological_sort(self.psmodel.jobs, metric='sdf')
        mrf = topological_sort(self.psmodel.jobs, metric='mrf')
        lrf = topological_sort(self.psmodel.jobs, metric='lrf')

        population = self.mcmc_sampling(ldf, proportion, 0, sampling)
        population += self.mcmc_sampling(sdf, proportion, 0, sampling)
        population += self.mcmc_sampling(mrf, proportion, 0, sampling)
        population += self.mcmc_sampling(lrf, proportion, 0, sampling)

        final_population = [EventList(psmodel=self.psmodel, jobs=p) for p in population] + [EventList(psmodel=self.psmodel) for _ in range(N-len(population))]
        return final_population

    def mcmc_sampling(self, permutation, num_sample, burn_in: int, spacing: int):

        if spacing == 0:
            return [permutation] * num_sample
        
        samples = []
        iterations = burn_in + num_sample * spacing

        for t in range(int(iterations)):
            idx = random.randint(1, len(self.psmodel.jobs)-3)
            if EventList.can_swap(permutation, idx):
                permutation[idx], permutation[idx+1] = permutation[idx+1], permutation[idx]
            if t >= burn_in and (t-burn_in)%spacing == 0:
                samples.append(permutation[:]) 

        return samples
    
    def set_parameters(self, N, limit, max_trials):
        self.params = Parameters(N, limit, max_trials)
        
    def employeed_phase(self):
        self.updates += 1
        num_sources = len(self.food_sources)
        step_choices = [1, 2, 3]
        step_weights = [0.9, 0.07, 0.03]
        random_steps = np.random.choice(step_choices, size=num_sources, p=step_weights)
        
        for i, (current_solution, steps) in enumerate(zip(self.food_sources, random_steps)):
            new_solution = current_solution.generate_new_local_solution(steps=steps)

            if new_solution.get_makespan() < current_solution.get_makespan():
                self.food_sources[i] = new_solution
                self.updates[i] = 0
                    
    def onlooker_phase(self):
        num_sources = len(self.food_sources)
        all_makespans = np.array([fs.get_makespan() for fs in self.food_sources])
        tournament_indices = np.random.randint(0, num_sources, size=(num_sources, 3))
        tournament_makespans = all_makespans[tournament_indices]
        winner_local_indices = np.argmin(tournament_makespans, axis=1)
        selected_indices = tournament_indices[np.arange(num_sources), winner_local_indices]
        for selected in selected_indices:
            self.updates[selected] += 1
            new_solution = self.food_sources[selected].generate_new_local_solution(steps=1)
            if new_solution.get_makespan() < self.food_sources[selected].get_makespan():
                self.food_sources[selected] = new_solution
                self.updates[selected] = 0

    def probability(self, cuotient):
        return 1/(1.1 + (3*cuotient)**3)
    
    def optimize(self, params: Parameters = None, mode="abc"):
        self.params = self.params if params is None else params
        p = self.params
        
        # --- Initialization ---
        if p.init_params is None:
            self.food_sources = [EventList(self.psmodel) for _ in range(p.N)]
        else:
            self.food_sources = self.init_population(p.N, p.init_params)
        
        self.updates = np.zeros(p.N, dtype=np.int32)
        
        all_makespans = np.array([s.get_makespan() for s in self.food_sources])
        best_idx = np.argmin(all_makespans)
        best_solution = copy.deepcopy(self.food_sources[best_idx])
        
        history = [best_solution]
        permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
        population_diversity = [positional_entropy(permutations)[1]]
        nscout_bees = [0]
        nunique_individuals = [calculate_unique_individuals(self.food_sources)]
        objective_evaluations = 0
        
        trials = 0
        while objective_evaluations < p.max_evaluations and best_solution.get_makespan() > self.psmodel.best_known:
            
            self.employeed_phase()
            self.onlooker_phase()
            
            objective_evaluations += 2 * p.N

            all_makespans = np.array([s.get_makespan() for s in self.food_sources])
            iter_best_idx = np.argmin(all_makespans)
            
            # Local search logic (optional)
            if p.l != -1 and (trials - 1) % p.l == 0:
                self.food_sources[iter_best_idx].shift_foward_search()
                all_makespans[iter_best_idx] = self.food_sources[iter_best_idx].get_makespan()

            # Check for new global best
            if all_makespans[iter_best_idx] < best_solution.get_makespan():
                best_solution = copy.deepcopy(self.food_sources[iter_best_idx])
                if p.l != -1:
                    best_solution.shift_foward_search()
                trials = 0
            else:
                trials = (trials + 1) % p.stagnation
            
            history.append(best_solution)

            # --- Scout Bee Phase) ---
            scout_indices = np.where(self.updates > p.limit)[0]
            if scout_indices.size > 0:
                if mode == "abc":
                    for i in scout_indices:
                        self.food_sources[i] = EventList(psmodel=self.psmodel)
                elif mode.startswith("gs-abc"):
                    prob_roll = np.random.random(len(scout_indices))
                    recombine_roll = np.random.random(len(scout_indices))
                    mutation_roll = np.random.random(len(scout_indices))

                    for j, i in enumerate(scout_indices):
                        if prob_roll[j] < self.probability(trials/p.stagnation):
                            self.food_sources[i] = EventList(psmodel=self.psmodel)
                            continue
                        
                        iter_best_solution = self.food_sources[iter_best_idx]
                        if recombine_roll[j] < 0.5:
                            self.food_sources[i] = iter_best_solution.recombine_solution(self.food_sources[i])
                        else:
                            self.food_sources[i] = self.food_sources[i].recombine_solution(iter_best_solution)

                        if mutation_roll[j] < p.mr:
                            self.food_sources[i] = self.food_sources[i].swap_new_solution(iterations=random.choices([1, 2, 3], weights=[0.75, 0.15, 0.1], k=1)[0])
                else:
                    raise ValueError(f'{mode} is not a valid mode')

                self.updates[scout_indices] = 0
            
            permutations = [[job.id for job in solution.jobs] for solution in self.food_sources]
            population_diversity.append(positional_entropy(permutations)[1])
            nunique_individuals.append(calculate_unique_individuals(self.food_sources))
            nscout_bees.append(len(scout_indices))

        self.population_divsersity = population_diversity
        self.nscout_bees = nscout_bees
        self.nunique_individuals = nunique_individuals
        self.history = history
        return self.history[-1]