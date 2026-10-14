"""RCPSP problem adapter for the ABC / GS-ABC optimizers."""

import copy
import random

from swarm.heuristics import topological_sort
from swarm.operators import adjacent_swap, multi_step_perturbation, recombine
from swarm.problem import ProjectSchedulingModel
from swarm.solution import Schedule


class RCPSPOperators:
    """Implements ``GSABCProblem[Schedule]``, ``SupportsInitialPopulation`` and ``SupportsLocalSearch``.

    ``heuristics_rate`` is the share of the initial colony seeded from the LDF, SDF, MRF and LRF priority
    rules, and ``sampling_rate`` the MCMC spacing between those samples (0 = copies of the rule's order).
    """

    PRIORITY_RULES = ("ldf", "sdf", "mrf", "lrf")

    def __init__(self, model: ProjectSchedulingModel, *, heuristics_rate: float = 0.0, sampling_rate: int = 25):
        self.model = model
        self.heuristics_rate = heuristics_rate
        self.sampling_rate = sampling_rate

    def random_solution(self) -> Schedule:
        return Schedule(psmodel=self.model)

    def neighbor(self, solution: Schedule, steps: int) -> Schedule:
        return multi_step_perturbation(solution, steps=steps)

    def recombine(self, a: Schedule, b: Schedule) -> Schedule:
        return recombine(a, b)

    def mutate(self, solution: Schedule, steps: int) -> Schedule:
        return adjacent_swap(solution, iterations=steps)

    def local_search(self, solution: Schedule) -> Schedule:
        improved = copy.deepcopy(solution)  # shift_forward() works in place
        improved.shift_forward()
        return improved

    def initial_population(self, n: int) -> list[Schedule]:
        per_rule = int(self.heuristics_rate * n * 0.25)
        orders = []
        for rule in self.PRIORITY_RULES:
            orders += self._mcmc_sampling(topological_sort(self.model.jobs, metric=rule), per_rule)
        schedules = [Schedule(psmodel=self.model, jobs=jobs) for jobs in orders]
        schedules += [self.random_solution() for _ in range(n - len(schedules))]
        return schedules

    def _mcmc_sampling(self, order: list, num_samples: int, burn_in: int = 0) -> list:
        spacing = self.sampling_rate
        if spacing == 0:
            return [order] * num_samples

        samples = []
        for t in range(int(burn_in + num_samples * spacing)):
            idx = random.randint(1, len(self.model.jobs) - 3)
            if Schedule.can_swap(order, idx):
                order[idx], order[idx + 1] = order[idx + 1], order[idx]
            if t >= burn_in and (t - burn_in) % spacing == 0:
                samples.append(order[:])
        return samples
