import json
import os
import random

from loguru import logger

from src.algorithms.h1 import Task, h1_algorithm
from src.algorithms.simulated_annealing import simulated_annealing
from src.settings import settings
from src.visualizer import Visualizer


def read_instance_data(file_path: str = None):
    directory = settings.instances_dir
    try:
        with open(os.path.join(directory, file_path), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file '{file_path}' not found. Exiting.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Could not decode JSON from '{file_path}'. Exiting."
        )
    return data

    return {
        "m": 8,
        "n": 5,
        "tasks": {
            "T1": {"work_amount": 100, "speed_up_factors": [1.0, 1.9, 2.0]},
            "T2": {"work_amount": 80, "speed_up_factors": [1.0, 1.5, 1.5, 1.5]},
            "T3": {"work_amount": 120, "speed_up_factors": [1.0, 2.0, 3.0, 3.9]},
            "T4": {"work_amount": 50, "speed_up_factors": [1.0, 1.5, 1.8, 2.0]},
            "T5": {"work_amount": 200, "speed_up_factors": [1.0, 1.7, 2.2, 2.5, 2.7]},
        },
    }


def save_schedule(schedule: dict, filename: str):
    directory = settings.schedules_dir
    filename = os.path.join(directory, filename)
    try:
        with open(filename, "w") as f:
            json.dump(schedule, f, indent=2)
            logger.info(f"Schedule saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save schedule to {filename}: {e}")


def pretty_print_schedule(algorithm_name: str, schedule: dict, makespan: float):
    print("-" * 20)
    print(f"{algorithm_name} makespan: {makespan:.5f}")

    s = schedule.copy()
    s.sort(key=lambda x: (x["task_id"], x["start_time"]))

    print("\nGenerated schedule log:")
    for entry in s:
        print(
            f"  Task {entry['task_id']}: "
            f"[{entry['start_time']:.4f} - {entry['end_time']:.4f}] "
            f"(Duration: {(entry['end_time'] - entry['start_time']):.4f}) "
            f"on {entry['num_processors']} processors"
        )
    print("-" * 20)


def unify_speedup_factors(speedup_factors: list[float], m: int):
    if len(speedup_factors) < m:
        last_val = speedup_factors[-1] if speedup_factors else 0
        speedup_factors.extend([last_val] * (m - len(speedup_factors)))
    elif len(speedup_factors) > m:
        speedup_factors = speedup_factors[:m]
    return speedup_factors


if __name__ == "__main__":
    # random.seed(42)

    data = read_instance_data("instance_4.json")

    m = data["m"]
    n = data["n"]
    task_data = data["tasks"]
    tasks = []

    for key, val in task_data.items():
        speedups = unify_speedup_factors(val["speed_up_factors"], m)
        task = Task(key, float(val["work_amount"]), speedups)
        tasks.append(task)

    makespan_h1, schedule_h1 = h1_algorithm(m=m, n=n, tasks=tasks)

    pretty_print_schedule("H1", schedule_h1, makespan_h1)

    save_schedule(schedule_h1, "h1_schedule.json")

    visualizer = Visualizer()
    visualizer.visualize_schedule_by_processor(
        schedule_h1,
        m,
        makespan_h1,
        "h1_schedule_by_processor.png",
        show=False,
    )
    simulated_annealing_schedule, simulated_annealing_makespan = simulated_annealing(
        instance_data=data,
        initial_temperature=50,
        final_temperature=5,
        cooling_rate=0.99,
        iterations_per_temperature=50,
    )

    pretty_print_schedule(
        "Simulated Annealing",
        simulated_annealing_schedule,
        simulated_annealing_makespan,
    )

    save_schedule(simulated_annealing_schedule, "simulated_annealing_schedule.json")

    visualizer.visualize_schedule_by_processor(
        simulated_annealing_schedule,
        data["m"],
        simulated_annealing_makespan,
        "simulated_annealing_schedule_by_processor.png",
        show=False,
    )
