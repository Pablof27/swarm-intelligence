from dataclasses import dataclass
from typing import List
import numpy as np
from itertools import count

@dataclass
class Job:
    id: int
    duration: int
    resources_needed: np.array
    sucessors: List[int]
    predecessors: List[int]
    start_time: int | None = None
    
    def isActiveAtTime(self, time: int) -> bool:
        return self.start_time is not None and self.start_time + self.duration > time and self.start_time <= time
    
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
    startTime: int

@dataclass
class Resources:
    resourcesList: List[Resource]
    resources: np.array
    
    def __init__(self, resourcesList: List[Resource]):
        self.resourcesList = resourcesList
        self.resources = np.array([resource.capacity for resource in resourcesList])
        
    def get_time_for_resources(self, jobs: List[Job], job: Job, base_time: int, end_time=np.Infinity) -> int:
        resources = self.resources
        for start_time in count(base_time):
            if start_time >= end_time:
                break
            time_found = True
            time = start_time
            while time < start_time + job.duration and time_found: 
                resources_used = np.zeros(len(resources), dtype=np.int32)
                active_jobs = [j for j in jobs if j.isActiveAtTime(time) or job == j]
                resources_used = sum(j.resources_needed for j in active_jobs)
                if np.any(resources_used > resources):
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
            for successor in job.sucessors:
                suc = next(j for j in jobs if j.id == successor)
                if job.start_time + job.duration > suc.start_time:
                    return False
        for t in range(jobs[-1].start_time):
            resources_used = np.zeros(len(self.resources.resources), dtype=np.int32)
            resources_used += sum(j.resources_needed if j.isActiveAtTime(t) else 0 for j in jobs)
            if np.any(resources_used > self.resources.resources):
                return False
        return True
    
    def times2jobs(self, times):
        jobs = []
        for i, t in enumerate(times):
            thisJob = next(filter(lambda j: j.id == i, self.jobs))
            job = Job(id=i, duration=thisJob.duration, resources_needed=thisJob.resources_needed, sucessors=thisJob.sucessors, predecessors=thisJob.predecessors, start_time=t)
            jobs.append(job)

        jobs = sorted(jobs, key=lambda j: j.start_time)
        return jobs
    
    @staticmethod
    def from_file(file_path: str) -> 'ProjectSchedulingModel':
        with open(file_path, "r") as file:
            lines = file.readlines()
        
        njobs = int(lines[5].split()[-1])
        renewable_resources_number = int(lines[8].split()[-2])
        nonrenewable_resources_number = int(lines[9].split()[-2])
        doble_constrain_resources_number = int(lines[10].split()[-2])
        nresources = renewable_resources_number + nonrenewable_resources_number + doble_constrain_resources_number
        
        succesors = []
        antecessors = [[] for i in range(njobs)]
        for i in range(18, 18 + njobs):
            line = lines[i].split()
            nsuccesor = int(line[2])
            it_succesors = [int(line[3 + j]) - 1 for j in range(nsuccesor)]
            succesors.append(it_succesors)
            for suc in it_succesors:
                antecessors[suc].append(i - 18)
                
        jobs = []
        for i in range(0, njobs):
            line = lines[22 + njobs + i].split()
            jobs.append(Job(id=i, duration=int(line[2]), resources_needed=np.array([int(line[3 + j]) for j in range(nresources)]), sucessors=succesors[i], predecessors=antecessors[i]))
        
        capacities_line = lines[2*njobs + 25].split()
        resourcesList = [Resource(id=i, capacity=int(capacities_line[i])) for i in range(nresources)]
        resources = Resources(resourcesList=resourcesList)
        name = file_path.split("/")[-1].split(".")[0]
        return ProjectSchedulingModel(name, jobs, resources)