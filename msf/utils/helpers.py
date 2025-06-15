import json
import os
from collections import defaultdict

from loguru import logger

from msf.config.settings import settings
from msf.utils.models import FinalSchedule


def read_schdeule_from_file(filename: str) -> FinalSchedule:
    directory = settings.schedules_dir
    filename = os.path.join(directory, filename)
    with open(filename, "r") as f:
        return FinalSchedule(**json.load(f))


def unify_speedup_factors(speedup_factors: list[float], m: int) -> list[float]:
    if len(speedup_factors) < m:
        last_val = speedup_factors[-1] if speedup_factors else 0
        speedup_factors.extend([last_val] * (m - len(speedup_factors)))
    elif len(speedup_factors) > m:
        speedup_factors = speedup_factors[:m]
    return speedup_factors


def pretty_print_schedule(algorithm_name: str, schedule: FinalSchedule) -> None:
    print("-" * 20)
    print(f"{algorithm_name} makespan: {schedule.makespan:.5f}")

    s = schedule.schedule.copy()
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


def read_instance_data(file_path: str | None = None) -> dict | None:
    directory = settings.instances_dir

    # if file_path is None:
    #     return None

    # try:
    #     with open(os.path.join(directory, file_path), "r") as f:
    #         data = json.load(f)
    # except FileNotFoundError:
    #     raise FileNotFoundError(f"Instance file '{file_path}' not found. Exiting.")
    # except json.JSONDecodeError:
    #     raise
    # return data

    return {
        "m": 8,
        "n": 5,
        "tasks": {
            "T1": {
                "work_amount": 100,
                "speed_up_factors": [1.0, 1.9, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5],
            },
            "T2": {
                "work_amount": 80,
                "speed_up_factors": [1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            },
            "T3": {
                "work_amount": 120,
                "speed_up_factors": [1.0, 2.0, 3.0, 3.9, 3.9, 3.9, 3.9, 3.9],
            },
            "T4": {
                "work_amount": 50,
                "speed_up_factors": [1.0, 1.5, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
            "T5": {
                "work_amount": 200,
                "speed_up_factors": [1.0, 1.7, 2.2, 2.5, 2.6, 2.6, 2.6, 2.6],
            },
        },
    }


def save_schedule(schedule: FinalSchedule, filename: str) -> None:
    directory = settings.schedules_dir
    filename = os.path.join(directory, filename)

    schedule_dict = {
        "num_processors": schedule.num_processors,
        "num_tasks": schedule.num_tasks,
        "makespan": schedule.makespan,
        "schedule": schedule.schedule,
    }

    try:
        with open(filename, "w") as f:
            json.dump(schedule_dict, f, indent=2)
            logger.info(f"Schedule saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save schedule to {filename}: {e}")


def group_intervals(records: list[dict]) -> list[dict]:
    grouped = defaultdict(int)

    for record in records:
        task_id = record["task_id"]
        start_time = record["start_time"]
        end_time = record["end_time"]
        num_proc = record["num_processors"]

        key = (task_id, start_time, end_time)
        grouped[key] += num_proc

    result = [
        {
            "task_id": task_id,
            "start_time": start,
            "end_time": end,
            "num_processors": num_proc,
        }
        for (task_id, start, end), num_proc in grouped.items()
    ]

    return result
