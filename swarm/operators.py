import random
import copy
from collections import deque

from swarm.solution import Schedule


def random_reinsert(schedule: Schedule) -> Schedule:
    current_jobs = schedule.jobs[:]

    possible_events = schedule.events[:-1]
    if not possible_events:
        return Schedule(schedule.psmodel, current_jobs)

    random_event = random.choice(possible_events)
    jobs_to_reinsert = [j for j in random_event.jobs if j.id != 0]

    if not jobs_to_reinsert:
        return Schedule(schedule.psmodel, current_jobs)

    jobs_to_reinsert_ids = {j.id for j in jobs_to_reinsert}
    fixed_jobs = [j for j in current_jobs if j.id not in jobs_to_reinsert_ids]

    for job_to_add in jobs_to_reinsert:
        pos_map = {job.id: i for i, job in enumerate(fixed_jobs)}

        pred_indices = [pos_map[p_id] for p_id in job_to_add.predecessors if p_id in pos_map]
        succ_indices = [pos_map[s_id] for s_id in job_to_add.successors if s_id in pos_map]

        lower_bound = max(pred_indices) if pred_indices else -1
        upper_bound = min(succ_indices) if succ_indices else len(fixed_jobs)

        if lower_bound + 1 > upper_bound:
            new_pos = lower_bound + 1
        else:
            new_pos = random.randint(lower_bound + 1, upper_bound)

        fixed_jobs.insert(new_pos, job_to_add)

    return Schedule(schedule.psmodel, fixed_jobs)


def adjacent_swap(schedule: Schedule, iterations: int = 1) -> Schedule:
    new_jobs = copy.deepcopy(schedule.jobs)
    for _ in range(iterations):
        idx = random.randint(1, len(schedule.jobs) - 3)
        if Schedule.can_swap(new_jobs, idx):
            new_jobs[idx], new_jobs[idx + 1] = new_jobs[idx + 1], new_jobs[idx]

    return Schedule(psmodel=schedule.psmodel, jobs=new_jobs)


def multi_step_perturbation(schedule: Schedule, steps: int = 1) -> Schedule:
    sol = schedule
    for _ in range(steps):
        sol = random_reinsert(sol)
    return sol


def recombine(schedule_a: Schedule, schedule_b: Schedule) -> Schedule:
    elite_jobs = []
    sorted_events = sorted(schedule_a.events, key=lambda e: len(e.jobs), reverse=True)
    for e in sorted_events:
        elite_jobs.extend(e.jobs)
        if len(elite_jobs) >= len(schedule_a.jobs) / 2:
            break

    elite_job_ids = {j.id for j in elite_jobs}

    other_jobs_to_add = [j for j in schedule_b.jobs if j.id not in elite_job_ids]

    candidate_queue = deque(other_jobs_to_add + elite_jobs)

    final_jobs = []
    final_job_ids = set()

    while candidate_queue:
        scheduled_this_pass = False
        for _ in range(len(candidate_queue)):
            job = candidate_queue.popleft()
            if set(job.predecessors).issubset(final_job_ids):
                final_jobs.append(job)
                final_job_ids.add(job.id)
                scheduled_this_pass = True
            else:
                candidate_queue.append(job)

        if not scheduled_this_pass:
            break

    return Schedule(schedule_a.psmodel, jobs=final_jobs)
