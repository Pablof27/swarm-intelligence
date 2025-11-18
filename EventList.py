from SchedulingProblem import Job, Event, Resources, ProjectSchedulingModel, List
import numpy as np
import random
from Utils import topological_sort, plot_solution
from dataclasses import dataclass
import copy
from dijkstar import Graph, find_path
from collections import deque

@dataclass
class EventList:
    
    psmodel: ProjectSchedulingModel
    events: List[Event]
    jobs: List[Job]
    
    def __init__(self, psmodel: ProjectSchedulingModel, jobs: List[Job] | None = None):
        self.events = []
        self.psmodel = psmodel
        
        initial_jobs = topological_sort(psmodel.jobs) if jobs is None else jobs
        self.jobs =  [copy.deepcopy(job) for job in initial_jobs]
        self.job_map = {job.id: job for job in self.jobs}
        self.create_event_list(psmodel.resources)
        
    def get_jobs(self) -> List[Job]:
        return [job for event in self.events for job in event.jobs]
    
    def get_makespan(self) -> int:
        return self.jobs[-1].start_time
    
    def plot(self, mapping=None):
        plot_solution(self, mapping=mapping)

    def critical_path(self):
        graph = Graph()
        for j in self.jobs:
            for sucessor_id in j.sucessors:
                sucessor = next(filter(lambda x: x.id == sucessor_id, self.jobs))
                graph.add_edge(j.id, sucessor_id, sucessor.start_time - j.start_time)
        critical_path = find_path(graph, 0, self.jobs[-1].id)
        self._critical_path = critical_path.nodes
        # self._critical_path.nodes = [n+1 for n in critical_path.nodes]

    def shift_foward_search(self):
        can_shift = True
        while can_shift:
            can_shift = False
            for job in self.jobs[1:]:
                old_start_time = job.start_time
                job.start_time = None
                predecessor_jobs = [self.job_map[p_id] for p_id in job.predecessors]
                last_predecessor = max(predecessor_jobs, key=lambda j: j.start_time + j.duration)
                predecessors_end_time = last_predecessor.start_time + last_predecessor.duration
                new_time = self.psmodel.resources.get_time_for_resources(self.jobs, job, predecessors_end_time, old_start_time)
                job.start_time = old_start_time
                if new_time < job.start_time:
                    can_shift = True
                    job.start_time = new_time

        self.jobs = sorted(self.jobs, key=lambda j: j.start_time)
        self.events = []
        self.add(self.jobs[0], start_time=0)
        current_time = 0
        for job in self.jobs[1:]:
            start_time = job.start_time
            if start_time == current_time:
                self.add(job, event_id=-1)
            if start_time > current_time:
                self.add(job, start_time=start_time)
                current_time = start_time
            if start_time < current_time:
                raise ValueError("Invalid start time")
                
    def create_event_list(self, resources: Resources):
        for job in self.jobs:
            job.start_time = None
        self.add(self.jobs[0], start_time=0)
        current_time = 0
        for job in self.jobs[1:]:
            predecessor_jobs = [self.job_map[p_id] for p_id in job.predecessors]
            last_predecessor = max(predecessor_jobs, key=lambda j: j.start_time + j.duration)
            predecessors_end_time = last_predecessor.start_time + last_predecessor.duration
            base_time = max(predecessors_end_time, current_time)
            start_time = resources.get_time_for_resources(self.jobs, job, base_time)
            if start_time == current_time:
                self.add(job, event_id=-1)
            if start_time > current_time:
                self.add(job, start_time=start_time)
                current_time = start_time
            if start_time < current_time:
                raise ValueError("Invalid start time")
        
    def add(self, job: Job, start_time: int | None = None, event_id: int | None = None):
        if event_id is None and start_time is None:
            raise ValueError("Either start_time or event_id must be provided")
        if event_id is not None and start_time is not None:
            raise ValueError("Only one of start_time or event_id can be provided")
        if event_id and event_id >= len(self.events):
            raise ValueError("Event ID does not exist")
        if start_time and start_time < self.events[-1].startTime:
            raise ValueError("Start time must be after the last event")
        
        if event_id is not None:
            self.events[event_id].jobs.append(job)
            job.start_time = self.events[event_id].startTime
        if start_time is not None:
            self.events.append(Event(len(self.events), [job], start_time))
            job.start_time = start_time
                        
    def random_local_solution(self) -> 'EventList':
        # Start with the current job order (shallow copy is sufficient here).
        current_jobs = self.jobs[:]
        
        # Select a random event and the jobs to be rescheduled.
        # Exclude the last event which might just be the dummy end job.
        possible_events = self.events[:-1]
        if not possible_events:
            return EventList(self.psmodel, current_jobs)
            
        random_event = random.choice(possible_events)
        # Don't move the dummy start job (id=0).
        jobs_to_reinsert = [j for j in random_event.jobs if j.id != 0]
        
        if not jobs_to_reinsert:
            return EventList(self.psmodel, current_jobs)
            
        # Create a list of jobs that are fixed for now.
        jobs_to_reinsert_ids = {j.id for j in jobs_to_reinsert}
        fixed_jobs = [j for j in current_jobs if j.id not in jobs_to_reinsert_ids]
        
        # Re-insert the chosen jobs one by one into the 'fixed_jobs' list.
        for job_to_add in jobs_to_reinsert:
            # For each job, find its valid insertion range in the current list.
            pos_map = {job.id: i for i, job in enumerate(fixed_jobs)}
            
            # Find the indices of predecessors and successors that are in the fixed list.
            pred_indices = [pos_map[p_id] for p_id in job_to_add.predecessors if p_id in pos_map]
            succ_indices = [pos_map[s_id] for s_id in job_to_add.sucessors if s_id in pos_map]

            # The job must be inserted after all its predecessors and before all its successors.
            lower_bound = max(pred_indices) if pred_indices else -1
            upper_bound = min(succ_indices) if succ_indices else len(fixed_jobs)
            
            # The valid range for insertion is (lower_bound, upper_bound].
            # randint is inclusive, so the range is [lower_bound + 1, upper_bound].
            if lower_bound + 1 > upper_bound:
                # No valid spot exists, insert at the earliest possible position.
                new_pos = lower_bound + 1
            else:
                 new_pos = random.randint(lower_bound + 1, upper_bound)
            
            fixed_jobs.insert(new_pos, job_to_add)

        return EventList(self.psmodel, fixed_jobs)

    def swap_new_solution(self, iterations=1) -> 'EventList':
        
        new_jobs = copy.deepcopy(self.jobs)
        for i in range(iterations):
            idx = random.randint(1, len(self.jobs)-3)
            if self.can_swap(new_jobs, idx):
                new_jobs[idx], new_jobs[idx+1] = new_jobs[idx+1], new_jobs[idx]
        
        return EventList(psmodel=self.psmodel, jobs=new_jobs)
                
    @staticmethod
    def can_swap(jobs, i) -> bool:
        a, b = jobs[i], jobs[i+1]

        if a.id in b.predecessors:
            return False
        
        return True
    
    def generate_new_local_solution(self, steps: int = 1) -> 'EventList':
        
        sol = self
        for _ in range(steps):
            sol = sol.random_local_solution()
            
        return sol

    # def recombine_solution(self, other: 'EventList') -> 'EventList':
    #     self_events = []
    #     selected_jobs = []
    #     sorted_events = sorted(self.events, key=lambda e: len(e.jobs), reverse=True)
    #     for e in sorted_events:
    #         self_events.append(e)
    #         selected_jobs.extend(e.jobs)
    #         if len(selected_jobs) >= len(self.jobs)/2:
    #             break
    #     self_events = sorted(self_events, key=lambda e: e.startTime)
    #     other_jobs = copy.deepcopy(list(filter(lambda j: j not in selected_jobs, other.jobs)))
    #     selected_jobs = []
    #     left_jobs_queue = []
    #     while len(self_events) > 0 or len(other_jobs) > 0 or len(left_jobs_queue) > 0:
    #         if len(left_jobs_queue) > 0:
    #             job = left_jobs_queue.pop(0)
    #             if set(job.predecessors).issubset([j.id for j in selected_jobs]):
    #                 selected_jobs.append(job)
    #             else:
    #                 left_jobs_queue.append(job)
    #         if len(other_jobs) > 0:
    #             job = other_jobs[0]
    #             if set(job.predecessors).issubset([j.id for j in selected_jobs]):
    #                 selected_jobs.append((other_jobs.pop(0)))
    #                 continue
    #         if len(self_events) == 0:
    #             continue
    #         event = self_events.pop(0)
    #         for j in event.jobs:
    #             if set(j.predecessors).issubset([j_.id for j_ in selected_jobs]):
    #                 selected_jobs.append(j)
    #             else:
    #                 left_jobs_queue.append(j)
        
    #     return EventList(self.psmodel, jobs=selected_jobs)

    def recombine_solution(self, other: 'EventList') -> 'EventList':
        """
        Recombines two solutions to create a new one.

        OPTIMIZATION:
        - This is a complete rewrite for performance and clarity.
        - Uses sets for fast filtering of jobs (O(1) per job).
        - Uses deques for efficient queue management.
        - The logic is simplified to a greedy scheduler: it tries to add jobs from a combined
          candidate pool into a new valid schedule, prioritizing the 'other' solution's order.
          This maintains the spirit of recombination while being much faster.
        """
        # 1. Select "elite" jobs from this solution's largest events.
        elite_jobs = []
        sorted_events = sorted(self.events, key=lambda e: len(e.jobs), reverse=True)
        for e in sorted_events:
            elite_jobs.extend(e.jobs)
            if len(elite_jobs) >= len(self.jobs) / 2:
                break
        
        elite_job_ids = {j.id for j in elite_jobs}

        # 2. Get jobs from the 'other' solution that are not in our elite set.
        # The job order from the other solution is preserved.
        other_jobs_to_add = [j for j in other.jobs if j.id not in elite_job_ids]

        # 3. Create a single pool of jobs to schedule, prioritizing 'other' solution's order,
        # then adding the elite jobs from 'self'.
        candidate_queue = deque(other_jobs_to_add + elite_jobs)
        
        final_jobs = []
        final_job_ids = set()
        
        # 4. Greedily build the new schedule by cycling through candidates.
        while candidate_queue:
            scheduled_this_pass = False
            # Iterate through all candidates currently in the queue.
            for _ in range(len(candidate_queue)):
                job = candidate_queue.popleft()
                # Use a set for fast predecessor check.
                if set(job.predecessors).issubset(final_job_ids):
                    final_jobs.append(job)
                    final_job_ids.add(job.id)
                    scheduled_this_pass = True
                else:
                    candidate_queue.append(job) # Put back at the end of the queue to wait.
            
            # If a full pass over the queue adds no new jobs, we are stuck.
            if not scheduled_this_pass:
                print(f"Warning: Recombination could not place {len(candidate_queue)} jobs due to dependency constraints.")
                break
                
        return EventList(self.psmodel, jobs=final_jobs)
    
    def generate_schedule_scheme(self) -> np.array:
        resources = self.psmodel.resources.resources
        end_time = self.events[-1].startTime
        schedules = [np.zeros((r, end_time), dtype=np.int32) - 1 for r in resources]
        jobs = [job for event in self.events for job in event.jobs]
        jobs = sorted(jobs, key=lambda x: x.duration, reverse=True)
        indexes = {i: {t: 0 for t in range(end_time)} for i in range(len(resources))}
        
        for job in jobs:
            for i, schedule in enumerate(schedules):
                
                # start = max((indexes[i][c] for c in range(job.start_time, job.start_time + job.duration)), default=0)
                # end = start + job.resources_needed[i]
                
                for col in range(end_time):
                    if job.isActiveAtTime(col):
                        start = indexes[i][col]
                        end = start + job.resources_needed[i]
                        schedule[start:end, col] = job.id
                        indexes[i][col] = end
                                                
        return schedules

    def __str__(self):
        sol_str = ""
        for event in self.events:
            sol_str += f"Event {event.id} at time {event.startTime}, jobs: {', '.join(str(job.id + 1) for job in event.jobs)}\n"
        return sol_str

    def __hash__(self):
        return id(self)