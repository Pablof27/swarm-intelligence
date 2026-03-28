from typing import List
import numpy as np
import copy

from swarm.problem import Job, Event, Resources, ProjectSchedulingModel
from swarm.heuristics import topological_sort


class Schedule:

    def __init__(self, psmodel: ProjectSchedulingModel, jobs: List[Job] | None = None):
        self.events: List[Event] = []
        self.psmodel = psmodel

        initial_jobs = topological_sort(psmodel.jobs) if jobs is None else jobs
        self.jobs = [copy.deepcopy(job) for job in initial_jobs]
        self.job_map = {job.id: job for job in self.jobs}
        self._create_event_list(psmodel.resources)

    def get_makespan(self) -> int:
        return self.jobs[-1].start_time

    def plot(self, mapping=None):
        from swarm.plotting import plot_solution
        plot_solution(self, mapping=mapping)

    def critical_path(self):
        from dijkstar import Graph, find_path

        graph = Graph()
        for j in self.jobs:
            for successor_id in j.successors:
                successor = next(s for s in self.jobs if s.id == successor_id)
                graph.add_edge(j.id, successor_id, successor.start_time - j.start_time)
        result = find_path(graph, 0, self.jobs[-1].id)
        self._critical_path = result.nodes

    def shift_forward(self):
        can_shift = True
        while can_shift:
            can_shift = False
            for job in self.jobs[1:]:
                old_start_time = job.start_time
                job.start_time = None
                predecessor_jobs = [self.job_map[p_id] for p_id in job.predecessors]
                last_predecessor = max(
                    predecessor_jobs, key=lambda j: j.start_time + j.duration
                )
                predecessors_end_time = (
                    last_predecessor.start_time + last_predecessor.duration
                )
                new_time = self.psmodel.resources.get_time_for_resources(
                    self.jobs, job, predecessors_end_time, old_start_time
                )
                job.start_time = old_start_time
                if new_time < job.start_time:
                    can_shift = True
                    job.start_time = new_time

        self.jobs = sorted(self.jobs, key=lambda j: j.start_time)
        self.events = []
        self._add(self.jobs[0], start_time=0)
        current_time = 0
        for job in self.jobs[1:]:
            start_time = job.start_time
            if start_time == current_time:
                self._add(job, event_id=-1)
            elif start_time > current_time:
                self._add(job, start_time=start_time)
                current_time = start_time
            else:
                raise ValueError("Invalid start time")

    def _create_event_list(self, resources: Resources):
        for job in self.jobs:
            job.start_time = None
        self._add(self.jobs[0], start_time=0)
        current_time = 0
        for job in self.jobs[1:]:
            predecessor_jobs = [self.job_map[p_id] for p_id in job.predecessors]
            last_predecessor = max(
                predecessor_jobs, key=lambda j: j.start_time + j.duration
            )
            predecessors_end_time = (
                last_predecessor.start_time + last_predecessor.duration
            )
            base_time = max(predecessors_end_time, current_time)
            start_time = resources.get_time_for_resources(self.jobs, job, base_time)
            if start_time == current_time:
                self._add(job, event_id=-1)
            elif start_time > current_time:
                self._add(job, start_time=start_time)
                current_time = start_time
            else:
                raise ValueError("Invalid start time")

    def _add(self, job: Job, start_time: int | None = None, event_id: int | None = None):
        if event_id is None and start_time is None:
            raise ValueError("Either start_time or event_id must be provided")
        if event_id is not None and start_time is not None:
            raise ValueError("Only one of start_time or event_id can be provided")
        if event_id and event_id >= len(self.events):
            raise ValueError("Event ID does not exist")
        if start_time and start_time < self.events[-1].start_time:
            raise ValueError("Start time must be after the last event")

        if event_id is not None:
            self.events[event_id].jobs.append(job)
            job.start_time = self.events[event_id].start_time
        if start_time is not None:
            self.events.append(Event(len(self.events), [job], start_time))
            job.start_time = start_time

    def generate_schedule_scheme(self) -> list:
        capacities = self.psmodel.resources.capacities
        end_time = self.events[-1].start_time
        schedules = [np.zeros((r, end_time), dtype=np.int32) - 1 for r in capacities]
        jobs = [job for event in self.events for job in event.jobs]
        jobs = sorted(jobs, key=lambda x: x.duration, reverse=True)
        indexes = {i: {t: 0 for t in range(end_time)} for i in range(len(capacities))}

        for job in jobs:
            for i, schedule in enumerate(schedules):
                for col in range(end_time):
                    if job.is_active_at_time(col):
                        start = indexes[i][col]
                        end = start + job.resources_needed[i]
                        schedule[start:end, col] = job.id
                        indexes[i][col] = end

        return schedules

    @staticmethod
    def can_swap(jobs, i) -> bool:
        a, b = jobs[i], jobs[i + 1]
        return a.id not in b.predecessors

    def __str__(self):
        sol_str = ""
        for event in self.events:
            sol_str += (
                f"Event {event.id} at time {event.start_time}, "
                f"jobs: {', '.join(str(job.id + 1) for job in event.jobs)}\n"
            )
        return sol_str

    def __hash__(self):
        return hash(tuple((job.id, job.start_time) for job in self.jobs))

    def __eq__(self, other):
        if not isinstance(other, Schedule):
            return NotImplemented
        return all(
            a.id == b.id and a.start_time == b.start_time
            for a, b in zip(self.jobs, other.jobs)
        )
