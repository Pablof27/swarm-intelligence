from typing import List, Any
import json
from dataclasses import dataclass, asdict, is_dataclass

from swarm.problem import ProjectSchedulingModel


@dataclass
class SolutionInfo:
    makespan: int
    start_times: List[int]


@dataclass
class ResultInfo:
    problem_id: str
    best: SolutionInfo
    best_history: List[int]
    population_diversity: List[float]
    unique_solutions: List[int]
    scout_bees: List[int]


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


def save_results(results: List[ResultInfo], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=4, cls=EnhancedJSONEncoder)


def load_results(filename: str) -> List[ResultInfo]:
    results: List[ResultInfo] = []
    with open(filename, "r") as f:
        data = json.load(f)
        for item_dict in data:
            best_solution = None
            if "best" in item_dict and item_dict["best"] is not None:
                best_solution = SolutionInfo(**item_dict["best"])

            results.append(
                ResultInfo(
                    problem_id=item_dict.get("problem_id", ""),
                    best=best_solution,
                    best_history=item_dict.get("best_history", []),
                    population_diversity=item_dict.get("population_diversity", []),
                    unique_solutions=item_dict.get("unique_solutions", []),
                    scout_bees=item_dict.get("scout_bees", []),
                )
            )
    return results


def load_problems(path: str) -> List[ProjectSchedulingModel]:
    return [
        ProjectSchedulingModel.from_file(f"{path}{i}_{j}.sm")
        for i in range(1, 49)
        for j in range(1, 11)
    ]


def load_best_known(dataset: int, path: str = "") -> List[int]:
    if dataset == 30:
        with open(f"{path}j30opt.sm", "r") as f:
            lines = f.readlines()
        return [int(line.split()[2]) for line in lines[22:502]]

    with open(f"{path}j{dataset}hrs.sm") as f:
        lines = f.readlines()
    return [int(line.split()[2]) for line in lines[4:484]]


def load_dataset(
    name: str, base_path: str = "Problems/"
) -> List[ProjectSchedulingModel]:
    dataset_size = int(name.replace("j", ""))
    path = f"{base_path}j{dataset_size}.sm/j{dataset_size}"
    problems = load_problems(path)
    best_known = load_best_known(dataset_size, path=base_path)
    for i, val in enumerate(best_known):
        if i < len(problems):
            problems[i].best_known = val
    return problems
