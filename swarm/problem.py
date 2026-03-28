from dataclasses import dataclass
from typing import List
import numpy as np
from itertools import count


@dataclass
class Job:
    id: int
    duration: int
    resources_needed: np.array
    successors: List[int]
    predecessors: List[int]
    start_time: int | None = None

    def is_active_at_time(self, time: int) -> bool:
        return (
            self.start_time is not None
            and self.start_time <= time
            and self.start_time + self.duration > time
        )

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Resource:
    id: int
    capacity: int


@dataclass
class Event:
    id: int
    jobs: List[Job]
    start_time: int


@dataclass
class Resources:
    resources_list: List[Resource]
    capacities: np.array

    def __init__(self, resources_list: List[Resource]):
        self.resources_list = resources_list
        self.capacities = np.array([r.capacity for r in resources_list])

    def get_time_for_resources(
        self, jobs: List[Job], job: Job, base_time: int, end_time=np.inf
    ) -> int:
        capacities = self.capacities
        for start_time in count(base_time):
            if start_time >= end_time:
                break
            time_found = True
            time = start_time
            while time < start_time + job.duration and time_found:
                resources_used = np.zeros(len(capacities), dtype=np.int32)
                active_jobs = [
                    j for j in jobs if j.is_active_at_time(time) or job == j
                ]
                resources_used = sum(j.resources_needed for j in active_jobs)
                if np.any(resources_used > capacities):
                    time_found = False
                time += 1
            if time_found:
                break
        return start_time


@dataclass
class ProjectSchedulingModel:
    name: str
    jobs: List[Job]
    resources: Resources
    best_known: int = 0

    def validate_solution(self, solution) -> bool:
        jobs = solution.jobs
        for job in jobs:
            for successor_id in job.successors:
                suc = next(j for j in jobs if j.id == successor_id)
                if job.start_time + job.duration > suc.start_time:
                    return False
        for t in range(jobs[-1].start_time):
            resources_used = np.zeros(len(self.resources.capacities), dtype=np.int32)
            for j in jobs:
                if j.is_active_at_time(t):
                    resources_used += j.resources_needed
            if np.any(resources_used > self.resources.capacities):
                return False
        return True

    def times2jobs(self, times):
        jobs = []
        for i, t in enumerate(times):
            source = next(j for j in self.jobs if j.id == i)
            job = Job(
                id=i,
                duration=source.duration,
                resources_needed=source.resources_needed,
                successors=source.successors,
                predecessors=source.predecessors,
                start_time=t,
            )
            jobs.append(job)
        return sorted(jobs, key=lambda j: j.start_time)

    @staticmethod
    def from_file(file_path: str) -> "ProjectSchedulingModel":
        with open(file_path, "r") as file:
            lines = file.readlines()

        njobs = int(lines[5].split()[-1])
        renewable = int(lines[8].split()[-2])
        nonrenewable = int(lines[9].split()[-2])
        doubly_constrained = int(lines[10].split()[-2])
        nresources = renewable + nonrenewable + doubly_constrained

        successors_list = []
        predecessors_list = [[] for _ in range(njobs)]
        for i in range(18, 18 + njobs):
            line = lines[i].split()
            nsuc = int(line[2])
            sucs = [int(line[3 + j]) - 1 for j in range(nsuc)]
            successors_list.append(sucs)
            for suc in sucs:
                predecessors_list[suc].append(i - 18)

        jobs = []
        for i in range(njobs):
            line = lines[22 + njobs + i].split()
            jobs.append(
                Job(
                    id=i,
                    duration=int(line[2]),
                    resources_needed=np.array(
                        [int(line[3 + j]) for j in range(nresources)]
                    ),
                    successors=successors_list[i],
                    predecessors=predecessors_list[i],
                )
            )

        capacities_line = lines[2 * njobs + 25].split()
        resources_list = [
            Resource(id=i, capacity=int(capacities_line[i])) for i in range(nresources)
        ]
        resources = Resources(resources_list=resources_list)
        name = file_path.split("/")[-1].split(".")[0]
        return ProjectSchedulingModel(name, jobs, resources)
