from typing import List
import random
import numpy as np

from swarm.problem import Job


METRICS = {
    "random": lambda queue: random.randint(0, len(queue) - 1),
    "ldf": lambda queue: max(enumerate(queue), key=lambda x: x[1].duration)[0],
    "sdf": lambda queue: min(enumerate(queue), key=lambda x: x[1].duration)[0],
    "mrf": lambda queue: max(enumerate(queue), key=lambda x: np.sum(x[1].resources_needed))[0],
    "lrf": lambda queue: min(enumerate(queue), key=lambda x: np.sum(x[1].resources_needed))[0],
}


def topological_sort(jobs: List[Job], metric: str = "random") -> List[Job]:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {list(METRICS.keys())}")

    select_func = METRICS[metric]

    in_degree = {job.id: 0 for job in jobs}
    graph = {job.id: [] for job in jobs}

    for job in jobs:
        for successor in job.successors:
            graph[job.id].append(successor)
            in_degree[successor] += 1

    queue = [job for job in jobs if in_degree[job.id] == 0]
    sorted_jobs = []

    while queue:
        job = queue.pop(select_func(queue))
        sorted_jobs.append(job)

        for successor in graph[job.id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                j = next(j for j in jobs if j.id == successor)
                queue.append(j)

    if len(sorted_jobs) != len(jobs):
        raise ValueError("The job graph has at least one cycle.")

    return sorted_jobs
