from Utils import loadProblems, loadBestKnown, save_results, ResultInfo, SolutionInfo, jobs2times
from ArtificialBeeColony import Parameters, ArtificialBeeColony
import os
import concurrent.futures
from tqdm import tqdm
import random

shutdown_event = False

def handle_signal(signum, frame):
    global shutdown_event
    print(f"Received signal {signum}. Initiating shutdown...")
    shutdown_event = True


PATHS = ["j30.sm/j30", "j60.sm/j60", "j90.sm/j90", "j120.sm/j120"]

def run_single_abc_optimization(problem_instance, params_obj, mode_str):
    abc = ArtificialBeeColony(psmodel=problem_instance)
    res = abc.optimize(params=params_obj, mode=mode_str)
    
    return ResultInfo(
        problem_id=problem_instance.name,
        best=SolutionInfo(
            makespan=res.get_makespan(),
            start_times=jobs2times(res.jobs)
        ),
        best_history=[sol.get_makespan() for sol in abc.history],
        population_diversity=abc.population_divsersity,
        unique_solutions=abc.nunique_individuals,
        scout_bees=abc.nscout_bees
    )

def experiment_parallel(problems_list, filename, base_params: Parameters, mode_str="abc", num_runs_per_problem=5, max_workers=None):

    all_results = []
    
    tasks_to_submit = []
    for p_instance in problems_list:
        for _ in range(num_runs_per_problem):
            tasks_to_submit.append((p_instance, base_params, mode_str))
            
    if not tasks_to_submit:
        print(f"No tasks to run for {filename}. Skipping.")
        return []

    print(f"Starting parallel execution for {filename} with {len(tasks_to_submit)} total runs.")
    print(f"Using {mode_str} with init params {base_params.init_params}.")
    
    if max_workers is None:
        max_workers = os.cpu_count()
        
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

            futures = [executor.submit(run_single_abc_optimization, p_inst, params_obj, m_str)
                    for p_inst, params_obj, m_str in tasks_to_submit]


            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Optimizing {filename}"):
                try:
                    result = future.result()  # Get the ResultInfo object from the completed future
                    all_results.append(result)
                    if shutdown_event:
                        print("Shutdown flag detected, stopping early.")
                        break
                except Exception as exc:

                    print(f"\nA task for {filename} generated an exception: {exc}")

    finally:
        if executor:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        print("Saving final results...")
        save_results(results=all_results, filename=filename)
        print(f"Results for {filename} saved successfully. Total results: {len(all_results)}.")

    
    return all_results

if __name__ == "__main__":
    print("Loading problems and best known solutions...")

    problems_data = {path.split(".")[0]: loadProblems(path) for path in PATHS}

    best_known_data = {f"j{i}": loadBestKnown(i) for i in [30, 60, 90, 120]}

    for key in problems_data.keys():
        if key in best_known_data:
            current_problems = problems_data[key]
            current_best_known = best_known_data[key]
            for i, best_val in enumerate(current_best_known):
                if i < len(current_problems):
                    current_problems[i].best_known = best_val
                else:
                    print(f"Warning: More best_known values than problems for {key}. Index {i} is out of bounds for problems list (length {len(current_problems)}).")
        else:
            print(f"Warning: No best_known data found for problem key {key}.")

    num_cores = os.cpu_count()
    print(f"Configured to use up to {num_cores} cores for parallel execution.")

    RES_PATH = "Results"

    n = 48
    problems_idx = {key: [i+random.randint(0, 480//n-1) for i in range(0, 480, 480//n)] for key in problems_data.keys()}
    print(problems_idx)
    problems_used = [problems_data[key][i] for key in problems_data.keys() for i in problems_idx[key]]
    print([problem.name for problem in problems_used])
    
    params_obj = Parameters(N=5*60, limit=60*2, max_trials=4*60*4, mr=0.15, l=25)
    experiment_parallel(
        problems_list=problems_used[48:48*2],
        filename=f"{RES_PATH}/j60-gs-abc.json",
        base_params=params_obj,
        mode_str="gs-abc",
        init_str="mcmc",
        num_runs_per_problem=5
    )