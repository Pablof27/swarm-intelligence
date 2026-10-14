"""Problem-agnostic Artificial Bee Colony (ABC) and Guided-Scout ABC (GS-ABC) with an ask/tell interface.

The optimizers never evaluate solutions themselves: ``ask()`` returns candidates, the caller evaluates them
and reports their objective values (minimisation) with ``tell()``. Everything that depends on the solution
representation lives in a problem object implementing the protocols below.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, Generic, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np

S = TypeVar("S")


@runtime_checkable
class ABCProblem(Protocol[S]):
    """Operators required by ABC. Operators must return new objects and never mutate their inputs."""

    def random_solution(self) -> S: ...

    def neighbor(self, solution: S, steps: int) -> S:
        """Neighbourhood move; ``steps`` is the perturbation strength (1 = short step, >1 = long step)."""
        ...


@runtime_checkable
class GSABCProblem(ABCProblem[S], Protocol[S]):
    """Additional operators required by the GS-ABC scouts."""

    def recombine(self, a: S, b: S) -> S: ...

    def mutate(self, solution: S, steps: int) -> S: ...


@runtime_checkable
class SupportsInitialPopulation(Protocol[S]):
    """Optional: build the initial colony (e.g. from heuristics) instead of drawing random solutions."""

    def initial_population(self, n: int) -> list[S]: ...


@runtime_checkable
class SupportsLocalSearch(Protocol[S]):
    """Optional: intensification applied to the iteration-best and to every new global best."""

    def local_search(self, solution: S) -> S: ...


class ABC(Generic[S]):
    """Artificial Bee Colony for minimisation, driven through ``ask()`` and ``tell()``.

    Each cycle runs the employed, onlooker, optional local search and scout phases. Every phase that needs
    evaluations is one ask/tell round (onlookers are one round per bee unless ``batch_onlookers`` is set).
    Transitions that need no evaluation happen inside ``tell()``, so ``best``, ``cycle``, etc. are always
    up to date, while candidate generation happens lazily inside ``ask()``.
    """

    STEP_CHOICES = (1, 2, 3)
    STEP_WEIGHTS = (0.90, 0.07, 0.03)

    def __init__(
        self,
        problem: ABCProblem[S],
        *,
        population_size: int = 10,
        limit: int = 100,
        stagnation: int = 1,
        local_search_interval: int = -1,
        batch_onlookers: bool = False,
        seed: int | None = None,
    ):
        if not isinstance(problem, ABCProblem):
            raise TypeError("problem must implement random_solution() and neighbor()")
        if local_search_interval != -1 and not isinstance(problem, SupportsLocalSearch):
            raise TypeError("local_search_interval requires the problem to implement local_search()")
        if population_size < 1:
            raise ValueError("population_size must be at least 1")
        if stagnation < 1:
            raise ValueError("stagnation must be at least 1")
        if local_search_interval != -1 and local_search_interval < 1:
            raise ValueError("local_search_interval must be -1 (disabled) or at least 1")

        if seed is not None:
            # Operators may draw from Python's and NumPy's global generators (the RCPSP ones do).
            random.seed(seed)
            np.random.seed(seed)

        self.problem = problem
        self.population_size = population_size
        self.limit = limit
        self.stagnation = stagnation
        self.local_search_interval = local_search_interval
        self.batch_onlookers = batch_onlookers
        self._local_search: Callable[[S], S] | None = (
            problem.local_search if isinstance(problem, SupportsLocalSearch) else None
        )

        self.population: list[S] = []
        self.fitness = np.empty(0)
        self.trials = np.zeros(population_size, dtype=np.int32)
        self.best: S | None = None
        self.best_fitness = np.inf
        self.cycle = 0
        self.n_evaluations = 0
        self.n_scouts = 0

        self._phase = "init"
        self._pending: list[S] | None = None
        self._stagnation_counter = 0
        self._onlooker_queue: deque[int] = deque()
        self._asked: list[int] = []
        self._iter_best = 0
        self._scouts = np.empty(0, dtype=np.intp)

    def ask(self) -> list[S]:
        """Return the candidates to evaluate next. Calling it again before ``tell()`` returns the same batch."""
        if self._pending is None:
            self._pending = getattr(self, f"_ask_{self._phase}")()
        return list(self._pending)

    def tell(self, fitness: Sequence[float]) -> None:
        """Report the objective values of the last ``ask()`` batch, in the same order."""
        if self._pending is None:
            raise RuntimeError("tell() called without a pending ask()")
        values = np.asarray(fitness, dtype=float)
        if values.shape != (len(self._pending),):
            raise ValueError(f"expected {len(self._pending)} fitness values, got shape {values.shape}")
        candidates, self._pending = self._pending, None
        self.n_evaluations += len(candidates)
        getattr(self, f"_tell_{self._phase}")(candidates, values)

    def _ask_init(self) -> list[S]:
        if isinstance(self.problem, SupportsInitialPopulation):
            return self.problem.initial_population(self.population_size)
        return [self.problem.random_solution() for _ in range(self.population_size)]

    def _tell_init(self, candidates: list[S], fitness: np.ndarray) -> None:
        self.population, self.fitness = list(candidates), fitness.copy()
        i = int(np.argmin(self.fitness))
        self.best, self.best_fitness = self.population[i], self.fitness[i]
        self._phase = "employed"

    def _ask_employed(self) -> list[S]:
        self.trials += 1
        steps = np.random.choice(self.STEP_CHOICES, size=self.population_size, p=self.STEP_WEIGHTS)
        return [self.problem.neighbor(s, int(k)) for s, k in zip(self.population, steps)]

    def _tell_employed(self, candidates: list[S], fitness: np.ndarray) -> None:
        for i, (candidate, f) in enumerate(zip(candidates, fitness)):
            self._greedy(i, candidate, f)
        self._phase = "onlooker"

    def _ask_onlooker(self) -> list[S]:
        if not self._onlooker_queue:
            self._onlooker_queue.extend(int(i) for i in self._tournament())
        k = len(self._onlooker_queue) if self.batch_onlookers else 1
        self._asked = [self._onlooker_queue.popleft() for _ in range(k)]
        return [self.problem.neighbor(self.population[i], 1) for i in self._asked]

    def _tell_onlooker(self, candidates: list[S], fitness: np.ndarray) -> None:
        for i, candidate, f in zip(self._asked, candidates, fitness):
            self.trials[i] += 1
            self._greedy(i, candidate, f)
        if not self._onlooker_queue:
            self._after_onlookers()

    def _after_onlookers(self) -> None:
        self._iter_best = int(np.argmin(self.fitness))
        if (
            self.local_search_interval != -1
            and (self._stagnation_counter - 1) % self.local_search_interval == 0
        ):
            self._phase = "local_search"
        else:
            self._update_best()

    def _ask_local_search(self) -> list[S]:
        assert self._local_search is not None
        return [self._local_search(self.population[self._iter_best])]

    def _tell_local_search(self, candidates: list[S], fitness: np.ndarray) -> None:
        i = self._iter_best
        if fitness[0] <= self.fitness[i]:
            self.population[i], self.fitness[i] = candidates[0], fitness[0]
        self._update_best()

    def _update_best(self) -> None:
        i = self._iter_best
        if self.fitness[i] >= self.best_fitness:
            self._stagnation_counter = (self._stagnation_counter + 1) % self.stagnation
            self._release_scouts()
            return
        self.best, self.best_fitness = self.population[i], self.fitness[i]
        self._stagnation_counter = 0
        if self.local_search_interval != -1:
            self._phase = "best_local_search"
        else:
            self._release_scouts()

    def _ask_best_local_search(self) -> list[S]:
        assert self._local_search is not None and self.best is not None
        return [self._local_search(self.best)]

    def _tell_best_local_search(self, candidates: list[S], fitness: np.ndarray) -> None:
        if fitness[0] <= self.best_fitness:
            self.best, self.best_fitness = candidates[0], fitness[0]
        self._release_scouts()

    def _release_scouts(self) -> None:
        self._scouts = np.flatnonzero(self.trials > self.limit)
        if self._scouts.size:
            self._phase = "scout"
        else:
            self._end_cycle()

    def _ask_scout(self) -> list[S]:
        return self._scout_candidates(self._scouts)

    def _scout_candidates(self, indices: np.ndarray) -> list[S]:
        return [self.problem.random_solution() for _ in indices]

    def _tell_scout(self, candidates: list[S], fitness: np.ndarray) -> None:
        for i, candidate, f in zip(self._scouts, candidates, fitness):
            self.population[i], self.fitness[i] = candidate, f
        self.trials[self._scouts] = 0
        self._end_cycle()

    def _end_cycle(self) -> None:
        self.n_scouts = int(self._scouts.size)
        self.cycle += 1
        self._phase = "employed"

    def _greedy(self, i: int, candidate: S, f: float) -> None:
        if f < self.fitness[i]:
            self.population[i], self.fitness[i] = candidate, f
            self.trials[i] = 0

    def _tournament(self, size: int = 3) -> np.ndarray:
        contenders = np.random.randint(0, self.population_size, size=(self.population_size, size))
        winners = np.argmin(self.fitness[contenders], axis=1)
        return contenders[np.arange(self.population_size), winners]


class GSABC(ABC[S]):
    """ABC whose scouts recombine abandoned sources with the iteration-best and optionally mutate them."""

    MUTATION_STEP_WEIGHTS = (0.75, 0.15, 0.10)

    def __init__(
        self,
        problem: GSABCProblem[S],
        *,
        mutation_rate: float = 0.1,
        population_size: int = 10,
        limit: int = 100,
        stagnation: int = 1,
        local_search_interval: int = -1,
        batch_onlookers: bool = False,
        seed: int | None = None,
    ):
        if not isinstance(problem, GSABCProblem):
            raise TypeError("GSABC requires the problem to implement recombine() and mutate()")
        super().__init__(
            problem,
            population_size=population_size,
            limit=limit,
            stagnation=stagnation,
            local_search_interval=local_search_interval,
            batch_onlookers=batch_onlookers,
            seed=seed,
        )
        self._operators = problem
        self.mutation_rate = mutation_rate

    def scout_probability(self) -> float:
        """Probability that a scout restarts from a random solution instead of recombining."""
        quotient = self._stagnation_counter / self.stagnation
        return 1 / (1.1 + (3 * quotient) ** 3)

    def _scout_candidates(self, indices: np.ndarray) -> list[S]:
        k = len(indices)
        restart_roll, order_roll, mutation_roll = (np.random.random(k) for _ in range(3))
        p_restart = self.scout_probability()
        ops = self._operators
        # Scouts are built in order and each one sees the scouts built before it.
        view = list(self.population)
        for j, i in enumerate(indices):
            if restart_roll[j] < p_restart:
                view[i] = ops.random_solution()
                continue
            elite = view[self._iter_best]
            if order_roll[j] < 0.5:
                child = ops.recombine(elite, view[i])
            else:
                child = ops.recombine(view[i], elite)
            if mutation_roll[j] < self.mutation_rate:
                steps = random.choices(self.STEP_CHOICES, weights=self.MUTATION_STEP_WEIGHTS, k=1)[0]
                child = ops.mutate(child, steps)
            view[i] = child
        return [view[i] for i in indices]
