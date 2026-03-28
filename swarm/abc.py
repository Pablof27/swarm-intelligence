import numpy as np
import random
import copy
from dataclasses import dataclass
from typing import List

from swarm.problem import ProjectSchedulingModel
from swarm.solution import Schedule
from swarm.heuristics import topological_sort
from swarm.operators import multi_step_perturbation, adjacent_swap, recombine
from swarm.metrics import positional_entropy, calculate_unique_individuals


@dataclass
class OptimizationResult:
    best_solution: Schedule
    best_makespan: int
    history: List[Schedule]
    population_diversity: List[float]
    unique_individuals: List[int]
    scout_bees: List[int]
    n_evaluations: int


@dataclass
class Parameters:
    population_size: int = 10
    limit: int = 100
    max_evaluations: int = 50000
    stagnation: int = 1
    mutation_rate: float = 0.1
    local_search_interval: int = -1
    heuristics_rate: float = 0.0
    sampling_rate: int = 25
    seed: int | None = None


class ABC:

    # Step probabilities for employed bee phase
    STEP_CHOICES = [1, 2, 3]
    STEP_WEIGHTS = [0.9, 0.07, 0.03]

    # Mutation step weights for gs-abc scout phase
    MUTATION_STEP_WEIGHTS = [0.75, 0.15, 0.1]

    def __init__(self, problem: ProjectSchedulingModel, **kwargs):
        self.problem = problem
        self.params = Parameters(**kwargs)

    def optimize(self, mode: str = "abc") -> OptimizationResult:
        p = self.params

        if p.seed is not None:
            random.seed(p.seed)
            np.random.seed(p.seed)

        # --- Initialization ---
        food_sources = self._init_population(p.population_size)
        updates = np.zeros(p.population_size, dtype=np.int32)

        all_makespans = np.array([s.get_makespan() for s in food_sources])
        best_idx = np.argmin(all_makespans)
        best_solution = copy.deepcopy(food_sources[best_idx])

        history = [best_solution]
        permutations = [[job.id for job in s.jobs] for s in food_sources]
        population_diversity = [positional_entropy(permutations)[1]]
        scout_bees_log = [0]
        unique_log = [calculate_unique_individuals(food_sources)]
        objective_evaluations = 0
        trials = 0

        while (
            objective_evaluations < p.max_evaluations
            and best_solution.get_makespan() > self.problem.best_known
        ):
            # --- Employed Bee Phase ---
            updates += 1
            step_choices = np.random.choice(
                self.STEP_CHOICES,
                size=p.population_size,
                p=self.STEP_WEIGHTS,
            )
            for i, (current, steps) in enumerate(zip(food_sources, step_choices)):
                new_solution = multi_step_perturbation(current, steps=steps)
                if new_solution.get_makespan() < current.get_makespan():
                    food_sources[i] = new_solution
                    updates[i] = 0

            # --- Onlooker Bee Phase ---
            all_makespans = np.array([fs.get_makespan() for fs in food_sources])
            tournament_indices = np.random.randint(
                0, p.population_size, size=(p.population_size, 3)
            )
            tournament_makespans = all_makespans[tournament_indices]
            winner_local = np.argmin(tournament_makespans, axis=1)
            selected_indices = tournament_indices[
                np.arange(p.population_size), winner_local
            ]
            for selected in selected_indices:
                updates[selected] += 1
                new_solution = multi_step_perturbation(food_sources[selected], steps=1)
                if new_solution.get_makespan() < food_sources[selected].get_makespan():
                    food_sources[selected] = new_solution
                    updates[selected] = 0

            objective_evaluations += 2 * p.population_size

            all_makespans = np.array([s.get_makespan() for s in food_sources])
            iter_best_idx = np.argmin(all_makespans)

            # Optional local search
            if (
                p.local_search_interval != -1
                and (trials - 1) % p.local_search_interval == 0
            ):
                food_sources[iter_best_idx].shift_forward()
                all_makespans[iter_best_idx] = food_sources[iter_best_idx].get_makespan()

            # Check for new global best
            if all_makespans[iter_best_idx] < best_solution.get_makespan():
                best_solution = copy.deepcopy(food_sources[iter_best_idx])
                if p.local_search_interval != -1:
                    best_solution.shift_forward()
                trials = 0
            else:
                trials = (trials + 1) % p.stagnation

            history.append(best_solution)

            # --- Scout Bee Phase ---
            scout_indices = np.where(updates > p.limit)[0]
            if scout_indices.size > 0:
                if mode == "abc":
                    for i in scout_indices:
                        food_sources[i] = Schedule(psmodel=self.problem)
                elif mode.startswith("gs-abc"):
                    prob_roll = np.random.random(len(scout_indices))
                    recombine_roll = np.random.random(len(scout_indices))
                    mutation_roll = np.random.random(len(scout_indices))

                    for j, i in enumerate(scout_indices):
                        if prob_roll[j] < self._scout_probability(trials / p.stagnation):
                            food_sources[i] = Schedule(psmodel=self.problem)
                            continue

                        iter_best = food_sources[iter_best_idx]
                        if recombine_roll[j] < 0.5:
                            food_sources[i] = recombine(iter_best, food_sources[i])
                        else:
                            food_sources[i] = recombine(food_sources[i], iter_best)

                        if mutation_roll[j] < p.mutation_rate:
                            steps = random.choices(
                                self.STEP_CHOICES,
                                weights=self.MUTATION_STEP_WEIGHTS,
                                k=1,
                            )[0]
                            food_sources[i] = adjacent_swap(food_sources[i], iterations=steps)
                else:
                    raise ValueError(f"'{mode}' is not a valid mode")

                updates[scout_indices] = 0

            # Track metrics
            permutations = [[job.id for job in s.jobs] for s in food_sources]
            population_diversity.append(positional_entropy(permutations)[1])
            unique_log.append(calculate_unique_individuals(food_sources))
            scout_bees_log.append(len(scout_indices))

        return OptimizationResult(
            best_solution=history[-1],
            best_makespan=history[-1].get_makespan(),
            history=history,
            population_diversity=population_diversity,
            unique_individuals=unique_log,
            scout_bees=scout_bees_log,
            n_evaluations=objective_evaluations,
        )

    def init_population(self, n: int) -> list[Schedule]:
        return self._init_population(n)

    def _init_population(self, n: int) -> list[Schedule]:
        p = self.params
        proportion = int(p.heuristics_rate * n * 0.25)
        spacing = p.sampling_rate

        ldf = topological_sort(self.problem.jobs, metric="ldf")
        sdf = topological_sort(self.problem.jobs, metric="sdf")
        mrf = topological_sort(self.problem.jobs, metric="mrf")
        lrf = topological_sort(self.problem.jobs, metric="lrf")

        population = self._mcmc_sampling(ldf, proportion, 0, spacing)
        population += self._mcmc_sampling(sdf, proportion, 0, spacing)
        population += self._mcmc_sampling(mrf, proportion, 0, spacing)
        population += self._mcmc_sampling(lrf, proportion, 0, spacing)

        schedules = [Schedule(psmodel=self.problem, jobs=p) for p in population]
        schedules += [Schedule(psmodel=self.problem) for _ in range(n - len(population))]
        return schedules

    def _mcmc_sampling(
        self, permutation, num_sample: int, burn_in: int, spacing: int
    ) -> list:
        if spacing == 0:
            return [permutation] * num_sample

        samples = []
        iterations = burn_in + num_sample * spacing

        for t in range(int(iterations)):
            idx = random.randint(1, len(self.problem.jobs) - 3)
            if Schedule.can_swap(permutation, idx):
                permutation[idx], permutation[idx + 1] = (
                    permutation[idx + 1],
                    permutation[idx],
                )
            if t >= burn_in and (t - burn_in) % spacing == 0:
                samples.append(permutation[:])

        return samples

    @staticmethod
    def _scout_probability(quotient: float) -> float:
        return 1 / (1.1 + (3 * quotient) ** 3)
