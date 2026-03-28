from typing import List
from math import log2
import numpy as np

from swarm.problem import ProjectSchedulingModel


def positional_entropy(permutations):
    if not permutations:
        return 0.0, 0.0

    permutations = np.array(permutations)
    m, n = permutations.shape
    max_entropy = log2(n)
    total_entropy = 0.0

    for i in range(n):
        counts = np.bincount(permutations[:, i], minlength=n + 1)[1:]
        probs = counts[counts > 0] / m
        entropy = -np.sum(probs * np.log2(probs))
        total_entropy += entropy

    average_entropy = total_entropy / n
    normalized_entropy = average_entropy / max_entropy if max_entropy > 0 else 0.0
    return average_entropy, normalized_entropy


def hamming_distance(p1, p2):
    return sum(a != b for a, b in zip(p1, p2))


def kendall_tau_distance(p1, p2):
    n = len(p1)
    index_map = {val: i for i, val in enumerate(p1)}
    mapped_p2 = [index_map[val] for val in p2]

    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped_p2[i] > mapped_p2[j]:
                inversions += 1
    return inversions


def average_pairwise_distance(permutations, metric="hamming"):
    m = len(permutations)
    if m < 2:
        return 0.0

    if metric == "hamming":
        distance_func = hamming_distance
    elif metric == "kendall":
        distance_func = kendall_tau_distance
    else:
        raise ValueError("Unsupported metric. Use 'hamming' or 'kendall'.")

    total_distance = 0
    count = 0
    for i in range(m - 1):
        for j in range(i + 1, m):
            total_distance += distance_func(permutations[i], permutations[j])
            count += 1

    return total_distance / count


def calculate_unique_individuals(population) -> int:
    unique = set()
    for individual in population:
        key = tuple(sorted((job.id, job.start_time) for job in individual.jobs))
        unique.add(key)
    return len(unique)


def normalize(data):
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def jobs2times(jobs):
    start_times = [0] * len(jobs)
    for j in jobs:
        start_times[j.id] = j.start_time
    return start_times


def arpd(results, problems: List[ProjectSchedulingModel]):
    deviations = {problem.name: [] for problem in problems}
    deviation_sum = 0
    for result in results:
        problem = next(p for p in problems if p.name == result.problem_id)
        dev = result.best.makespan - problem.best_known
        deviations[problem.name].append(dev)
        deviation_sum += dev

    return deviation_sum / len(deviations), deviations
