# Swarm Intelligence for Resource-Constrained Project Scheduling

An Artificial Bee Colony (ABC) optimization algorithm for solving the **Resource-Constrained Project Scheduling Problem (RCPSP)**, developed as part of a Master's Thesis in Artificial Intelligence.

The algorithm includes a standard ABC variant and a guided-scout extension (GS-ABC) that uses recombination and mutation operators in the scout bee phase to improve convergence.

## Quick Start

```python
from swarm import ABC, load_dataset

problems = load_dataset("j30")
result = ABC(problems[0], population_size=10, max_evaluations=50000).optimize()

print(f"Makespan: {result.best_makespan}")
result.best_solution.plot()
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

```python
from swarm import ABC

result = ABC(
    problems[121],
    population_size=4,
    limit=125,
    max_evaluations=50000,
    heuristics_rate=1.0,   # proportion of heuristic-based initialization
    sampling_rate=0,        # MCMC spacing (0 = no sampling)
    seed=42,                # reproducible results
).optimize(mode="abc")      # "abc" or "gs-abc"
```

The returned `OptimizationResult` contains:

| Field                  | Description                              |
|------------------------|------------------------------------------|
| `best_solution`        | Best `Schedule` found                    |
| `best_makespan`        | Makespan of the best solution            |
| `history`              | Best solution per cycle                  |
| `population_diversity` | Normalized positional entropy per cycle  |
| `unique_individuals`   | Count of unique solutions per cycle      |
| `scout_bees`           | Scout activations per cycle              |
| `n_evaluations`        | Total objective function evaluations     |

### Parameters

| Parameter            | Default | Description                                    |
|----------------------|---------|------------------------------------------------|
| `population_size`    | 10      | Number of food sources                         |
| `limit`              | 100     | Abandonment threshold for scout bees           |
| `max_evaluations`    | 50000   | Stopping criterion                             |
| `stagnation`         | 1       | Stagnation tracking window                     |
| `mutation_rate`      | 0.1     | Mutation probability (GS-ABC scout phase)      |
| `local_search_interval` | -1   | Forward-shift interval (-1 = disabled)         |
| `heuristics_rate`    | 0.0     | Proportion of heuristic-seeded init population |
| `sampling_rate`      | 25      | MCMC spacing between samples                   |
| `seed`               | None    | Random seed for reproducibility                |

### Visualization

```python
# Convergence plot
import matplotlib.pyplot as plt
plt.plot([s.get_makespan() for s in result.history])
plt.xlabel("Cycles")
plt.ylabel("Makespan")
plt.show()

# Gantt-style resource schedule
result.best_solution.plot()
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
permutations = [[job.id for job in s.jobs] for s in population]
avg_entropy, normalized = positional_entropy(permutations)

# Average Relative Percent Deviation
avg_dev, deviations = arpd(results, problems)
```

## Package Structure

```
swarm/
├── __init__.py      # Public API
├── problem.py       # Job, Resource, ProjectSchedulingModel
├── solution.py      # Schedule (feasible schedule representation)
├── operators.py     # random_reinsert, adjacent_swap, recombine, ...
├── heuristics.py    # Topological sort with priority rules (LDF, SDF, MRF, LRF)
├── abc.py           # ABC / GS-ABC optimizer
├── metrics.py       # Entropy, distances, ARPD
├── io.py            # Dataset loading and result serialization
└── plotting.py      # Gantt-style schedule visualization
```

## Algorithm Overview

The ABC algorithm models three types of foraging behavior:

1. **Employed bees** explore neighborhoods of current solutions via random event reinsertion (with configurable multi-step perturbation).
2. **Onlooker bees** select promising solutions through tournament selection and apply local search.
3. **Scout bees** replace abandoned solutions. In standard ABC, scouts generate random solutions. In GS-ABC, scouts use recombination with the iteration-best solution and optional mutation.

Population initialization uses four scheduling heuristics (Longest Duration First, Shortest Duration First, Most Resources First, Least Resources First) with MCMC-based sampling to control diversity.

## License

MIT
