# Swarm Intelligence for Resource-Constrained Project Scheduling

An Artificial Bee Colony (ABC) optimization algorithm for solving the **Resource-Constrained Project Scheduling Problem (RCPSP)**, developed as part of a Master's Thesis in Artificial Intelligence.

The algorithm includes a standard ABC variant and a guided-scout extension (GS-ABC) that uses recombination and mutation operators in the scout bee phase to improve convergence. Both optimizers are problem-agnostic and expose an **ask/tell** interface: the RCPSP-specific operators live in `RCPSPOperators`, and any other problem can be plugged in by implementing a few methods (see [Solving Other Problems](#solving-other-problems)).

## Quick Start

```python
from swarm import ABC, RCPSPOperators, load_dataset

problem = load_dataset("j30")[0]
optimizer = ABC(RCPSPOperators(problem), population_size=10)

while optimizer.n_evaluations < 50000 and optimizer.best_fitness > problem.best_known:
    schedules = optimizer.ask()
    optimizer.tell([s.get_makespan() for s in schedules])

print(f"Makespan: {optimizer.best.get_makespan()}")
optimizer.best.plot()
```

## Installation

```bash
git clone https://github.com/Pablof27/swarm-intelligence.git
cd swarm-intelligence
pip install numpy matplotlib dijkstar
```

## Benchmarks

The algorithm is evaluated on the standard [PSPLib](http://www.om-db.wi.tum.de/psplib/) benchmark sets:

| Dataset | Jobs | Instances |
|---------|------|-----------|
| j30     | 30   | 480       |
| j60     | 60   | 480       |
| j90     | 90   | 480       |
| j120    | 120  | 480       |

Place the benchmark files under `Problems/` following the existing directory structure. Best-known solutions are loaded automatically by `load_dataset()`.

## Usage

### Loading Problems

```python
from swarm import load_dataset

# Loads all 480 instances with best-known solutions attached
problems = load_dataset("j30")

print(problems[0].name)        # e.g. "j301_1"
print(problems[0].best_known)  # optimal/best-known makespan
```

### Running the Optimizer

The optimizers never evaluate solutions themselves. `ask()` returns a batch of candidates; you evaluate them and report their objective values (lower is better) with `tell()`, in the same order. The loop, the stopping criterion and any logging stay in your code.

```python
from swarm import GSABC, RCPSPOperators

problem = problems[121]
operators = RCPSPOperators(
    problem,
    heuristics_rate=1.0,   # proportion of heuristic-based initialization
    sampling_rate=0,       # MCMC spacing (0 = no sampling)
)
optimizer = GSABC(         # or ABC(...) for the standard variant
    operators,
    population_size=4,
    limit=125,
    stagnation=5,
    seed=42,               # reproducible results
)

while optimizer.n_evaluations < 50000 and optimizer.best_fitness > problem.best_known:
    schedules = optimizer.ask()
    optimizer.tell([s.get_makespan() for s in schedules])
```

For the RCPSP, `ask()` returns `Schedule` objects that are already decoded, so evaluating a candidate only means reading its makespan.

The optimizer state can be inspected at any time:

| Attribute       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `best`          | Best solution found so far                                                  |
| `best_fitness`  | Objective value (makespan) of `best`                                        |
| `population`    | Current food sources                                                        |
| `fitness`       | Objective values of `population`                                            |
| `trials`        | Failed improvement attempts per food source (abandoned above `limit`)       |
| `cycle`         | Completed ABC cycles                                                        |
| `n_evaluations` | Candidates evaluated so far, including initialization, local search and scouts |
| `n_scouts`      | Scout bees released in the last completed cycle                             |

### Parameters

`ABC` and `GSABC`:

| Parameter               | Default | Description                                                          |
|-------------------------|---------|----------------------------------------------------------------------|
| `population_size`       | 10      | Number of food sources                                               |
| `limit`                 | 100     | Abandonment threshold for scout bees                                 |
| `stagnation`            | 1       | Stagnation tracking window                                           |
| `local_search_interval` | -1      | Local search interval (-1 = disabled)                                |
| `batch_onlookers`       | False   | Ask for all onlooker candidates in one batch instead of one per bee  |
| `seed`                  | None    | Seeds Python's and NumPy's global random generators                  |
| `mutation_rate`         | 0.1     | Mutation probability in the scout phase (`GSABC` only)               |

`RCPSPOperators`:

| Parameter         | Default | Description                                    |
|-------------------|---------|------------------------------------------------|
| `heuristics_rate` | 0.0     | Proportion of heuristic-seeded init population |
| `sampling_rate`   | 25      | MCMC spacing between samples                   |

By default every onlooker bee is its own ask/tell round, so a food source chosen by several onlookers is improved sequentially, as in the canonical algorithm. `batch_onlookers=True` evaluates all the onlookers of a cycle in one batch, which suits parallel evaluation but slightly changes the search trajectory.

### Visualization

Since the loop is yours, record whatever you need while optimizing:

```python
import matplotlib.pyplot as plt

optimizer = GSABC(RCPSPOperators(problem), seed=42)
evaluations, makespans = [], []
while optimizer.n_evaluations < 50000 and optimizer.best_fitness > problem.best_known:
    schedules = optimizer.ask()
    optimizer.tell([s.get_makespan() for s in schedules])
    evaluations.append(optimizer.n_evaluations)
    makespans.append(optimizer.best_fitness)

# Convergence plot
plt.plot(evaluations, makespans)
plt.xlabel("Evaluations")
plt.ylabel("Makespan")
plt.show()

# Gantt-style resource schedule
optimizer.best.plot()
```

### Scheduling Heuristics

```python
from swarm import Schedule, topological_sort

# Available: "ldf", "sdf", "mrf", "lrf", "random"
jobs = topological_sort(problem.jobs, metric="ldf")
schedule = Schedule(psmodel=problem, jobs=jobs)
print(schedule.get_makespan())
```

### Analysis Metrics

```python
from swarm import positional_entropy, arpd

# Population diversity
permutations = [[job.id for job in s.jobs] for s in optimizer.population]
avg_entropy, normalized = positional_entropy(permutations)

# Average Relative Percent Deviation
avg_dev, deviations = arpd(results, problems)
```

### Solving Other Problems

`ABC` and `GSABC` only touch solutions through the problem object you pass them, so any representation works. Implement the methods of the corresponding protocol (exported as `ABCProblem`, `GSABCProblem`, `SupportsInitialPopulation` and `SupportsLocalSearch` for type hints):

| Method                      | Needed by                     | Purpose                                                |
|-----------------------------|-------------------------------|--------------------------------------------------------|
| `random_solution()`         | `ABC`, `GSABC`                | Random solution for initialization and scout bees      |
| `neighbor(solution, steps)` | `ABC`, `GSABC`                | Neighborhood move of strength `steps` (1 = short step) |
| `recombine(a, b)`           | `GSABC`                       | Combine an abandoned source with the iteration-best    |
| `mutate(solution, steps)`   | `GSABC`                       | Perturb a recombined scout                             |
| `initial_population(n)`     | optional                      | Custom initial colony (default: `n` random solutions)  |
| `local_search(solution)`    | `local_search_interval != -1` | Refine the best solutions                              |

Operators must return new objects and never modify their inputs. A travelling salesman problem, for example:

```python
import random
import numpy as np
from swarm import ABC

class TSP:
    def __init__(self, cities):
        self.cities = np.asarray(cities)

    def length(self, tour):
        points = self.cities[tour]
        return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())

    def random_solution(self):
        return np.random.permutation(len(self.cities))

    def neighbor(self, tour, steps):
        tour = tour.copy()
        for _ in range(steps):  # one step = one random 2-opt reversal
            i, j = sorted(random.sample(range(len(tour)), 2))
            tour[i : j + 1] = tour[i : j + 1][::-1]
        return tour

tsp = TSP(np.random.default_rng(0).random((50, 2)))
optimizer = ABC(tsp, population_size=20, limit=60, seed=0)
while optimizer.n_evaluations < 20000:
    tours = optimizer.ask()
    optimizer.tell([tsp.length(tour) for tour in tours])

print(optimizer.best_fitness)
```

Adding `recombine` and `mutate` is enough to solve the same problem with `GSABC`.

## Package Structure

```
swarm/
├── __init__.py      # Public API
├── problem.py       # Job, Resource, ProjectSchedulingModel
├── solution.py      # Schedule (feasible schedule representation)
├── operators.py     # random_reinsert, adjacent_swap, recombine, ...
├── heuristics.py    # Topological sort with priority rules (LDF, SDF, MRF, LRF)
├── abc.py           # Problem-agnostic ABC / GS-ABC optimizers (ask/tell) and problem protocols
├── rcpsp.py         # RCPSPOperators: RCPSP operators, heuristic initialization and local search
├── metrics.py       # Entropy, distances, ARPD
├── io.py            # Dataset loading and result serialization
└── plotting.py      # Gantt-style schedule visualization
```

## Algorithm Overview

The ABC algorithm models three types of foraging behavior:

1. **Employed bees** explore the neighborhood of every food source. Most moves are short (one step) and some are longer (two or three steps); for the RCPSP a step is a random event reinsertion.
2. **Onlooker bees** select promising solutions through tournament selection and explore their neighborhood with short moves.
3. **Scout bees** replace abandoned solutions. In standard ABC, scouts generate random solutions. In GS-ABC, scouts use recombination with the iteration-best solution and optional mutation.

When `local_search_interval` is set, a local search (for the RCPSP, shifting activities to their earliest feasible start) refines the iteration-best solution and every new global best.

Each phase that needs evaluations is one ask/tell round: one batch for the employed bees, one round per onlooker bee (or a single batch with `batch_onlookers=True`), and one batch each for the local search and the scouts when they run.

For the RCPSP, `RCPSPOperators` builds the initial population from four scheduling heuristics (Longest Duration First, Shortest Duration First, Most Resources First, Least Resources First) with MCMC-based sampling to control diversity.

## License

MIT
