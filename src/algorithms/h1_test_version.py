import heapq  # Can be useful for efficiently finding minimums
import json
import math
import random  # For distinct colors
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.settings import settings
from src.visualizer import Visualizer


class CandidateAction(str, Enum):
    START = "start"
    ADD = "add"


class Task:
    def __init__(self, id: str, work_amount: float, speed_up_factors: list[float]):
        self.id = id
        self.initial_work_amount = work_amount
        self.work_amount_left = work_amount
        self.speed_up_factors = speed_up_factors

    def __str__(self):
        return (
            f"Task(id={self.id!r}, "
            f"initial_work_amount={self.initial_work_amount:.5f}, "
            f"work_amount_left={self.work_amount_left:.5f}, "
            f"speed_up_factors={self.speed_up_factors})"
        )

    # Add comparison methods for tie-breaking if needed (e.g., by ID)
    def __lt__(self, other):
        return self.id < other.id

    __repr__ = __str__


class CandidateAllocation:
    def __init__(
        self,
        amount_of_work: float,
        n_processors: int,
        priority: int,
        action: CandidateAction,
        task: Task,
    ):
        self.amount_of_work = amount_of_work
        self.n_processors = n_processors
        self.priority = priority
        self.action = action
        self.task = task


def get_candidate_allocations(
    queue: dict[str, Task], running_list: dict[str, tuple[Task, int]]
) -> list[CandidateAllocation]:
    """
    Returns a list of candidate allocations for the waiting tasks and running tasks.
    """
    candidates = []

    for task_id, task in queue.items():
        candidates.append(
            CandidateAllocation(
                amount_of_work=task.work_amount_left,
                n_processors=1,
                priority=0,
                action=CandidateAction.START,
                task=task,
            )
        )

    for task_id, (task, p) in running_list.items():
        if (
            p < m
            and task.speed_up_factors[p]
            > task.speed_up_factors[p - 1] + settings.epsilon
        ):
            candidates.append(
                CandidateAllocation(
                    amount_of_work=task.work_amount_left,
                    n_processors=p + 1,
                    priority=1,
                    action=CandidateAction.ADD,
                    task=task,
                )
            )

    return candidates


def greedy_scheduling(m: int, n: int, tasks: list[Task]):
    """
    Greedy algorithm for malleable job scheduling (minimize makespan).
    Uses a Shortest Remaining Time First (SRTF) approach.
    Returns the makespan and a detailed schedule log.
    """
    current_time = 0.0
    free_processors = m
    waiting_tasks = {task.id: task for task in tasks}
    running_tasks = {}
    completed_count = 0
    schedule_log = []
    active_intervals = {}

    while completed_count < n:
        while free_processors > 0:
            if not (
                candidates := get_candidate_allocations(waiting_tasks, running_tasks)
            ):
                break

            # Select and perform best allocation
            candidates.sort(key=lambda x: (x.amount_of_work, x.priority, x.task.id))

            best_candidate = candidates[0]

            if best_candidate.action == CandidateAction.START:
                task_to_start = waiting_tasks.pop(best_candidate.task.id)
                running_tasks[task_to_start.id] = (task_to_start, 1)
                free_processors -= 1
                active_intervals[task_to_start.id] = {
                    "start_time": current_time,
                    "num_processors": best_candidate.n_processors,
                }
            else:  # ADD
                task_to_add_to, p = running_tasks[best_candidate.task.id]
                if task_to_add_to.work_amount_left > settings.epsilon:
                    old_interval = active_intervals.pop(best_candidate.task.id)
                    if old_interval["start_time"] < current_time:
                        schedule_log.append(
                            {
                                "task_id": best_candidate.task.id,
                                "start_time": old_interval["start_time"],
                                "end_time": current_time,
                                "num_processors": old_interval["num_processors"],
                            }
                        )
                    active_intervals[best_candidate.task.id] = {
                        "start_time": current_time,
                        "num_processors": best_candidate.n_processors,
                    }
                running_tasks[best_candidate.task.id] = (task_to_add_to, p + 1)
                free_processors -= 1

        # Calculate time to next completion
        min_delta_t = float("inf")
        tasks_to_complete = []

        for task_id, (task, p) in running_tasks.items():
            time_to_finish = task.work_amount_left / task.speed_up_factors[p - 1]
            if time_to_finish < min_delta_t + settings.epsilon:
                if time_to_finish < min_delta_t - settings.epsilon:
                    min_delta_t = time_to_finish
                    tasks_to_complete = [(time_to_finish, task_id)]
                else:
                    tasks_to_complete.append((time_to_finish, task_id))

        if min_delta_t == float("inf"):
            # Check for tasks that are already done
            finished_tasks = {
                tid
                for tid, (t, p) in running_tasks.items()
                if t.work_amount_left <= settings.epsilon
            }
            if finished_tasks:
                min_delta_t = 0
                tasks_to_complete = [(0.0, tid) for tid in finished_tasks]
            else:
                print(f"Error: No task progress possible at time {current_time}")
                break

        min_delta_t = max(0.0, min_delta_t)
        current_time += min_delta_t

        # Update work and handle completions
        if min_delta_t > 0:
            for task_id, (task, p) in running_tasks.items():
                work_done = task.speed_up_factors[p - 1] * min_delta_t
                task.work_amount_left -= work_done

        # Process completions - create a copy of items to avoid modification during iteration
        for task_id, (task, p) in list(running_tasks.items()):
            if task.work_amount_left <= settings.epsilon:
                if task_id in active_intervals:
                    interval = active_intervals.pop(task_id)
                    # Only add interval if it has non-zero duration
                    if interval["start_time"] < current_time:
                        schedule_log.append(
                            {
                                "task_id": task_id,
                                "start_time": interval["start_time"],
                                "end_time": current_time,
                                "num_processors": interval["num_processors"],
                            }
                        )
                free_processors += p
                completed_count += 1
                task.work_amount_left = 0.0
                del running_tasks[task_id]

    # Final check for any remaining active intervals
    if active_intervals:
        print("Warning: Some tasks still have active intervals after completion")
        for task_id, interval in active_intervals.items():
            if interval["start_time"] < current_time:
                schedule_log.append(
                    {
                        "task_id": task_id,
                        "start_time": interval["start_time"],
                        "end_time": current_time,
                        "num_processors": interval["num_processors"],
                    }
                )

    schedule_log.sort(key=lambda x: (x["start_time"], x["task_id"]))
    return current_time, schedule_log


# --- Main execution part ---
if __name__ == "__main__":
    instance_file = "_instances/instance_1.json"
    try:
        # data = json.load(open(instance_file))

        # data = {
        #     "n": 3,
        #     "m": 4,  # Total system processors
        #     "tasks": {
        #         "task_1": {
        #             "work_amount": 100.0,
        #             "speed_up_factors": [1.0, 1.8, 2.5, 2.8],
        #         },  # Defined for 1,2,3,4 P
        #         "task_2": {
        #             "work_amount": 120.0,
        #             "speed_up_factors": [1.0, 1.9, 2.6, 2.6],
        #         },  # Defined for 1,2,3 P
        #         "task_3": {
        #             "work_amount": 80.0,
        #             "speed_up_factors": [1.0, 2.0, 2.0, 2.0],
        #         },  # Defined for 1,2 P
        #     },
        # }

        data = {
            "m": 8,
            "n": 5,
            "tasks": {
                "T1": {"work_amount": 100, "speed_up_factors": [1.0, 1.9, 2.0, 2.0]},
                "T2": {"work_amount": 80, "speed_up_factors": [1.0, 1.5, 1.5, 1.5]},
                "T3": {"work_amount": 120, "speed_up_factors": [1.0, 2.0, 3.0, 3.9]},
                "T4": {"work_amount": 50, "speed_up_factors": [1.0, 1.5, 1.8, 2.0]},
                "T5": {"work_amount": 200, "speed_up_factors": [1.0, 1.7, 2.2, 2.5]},
            },
        }

    except FileNotFoundError:
        print(f"Error: Instance file '{instance_file}' not found. Exiting.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{instance_file}'. Exiting.")
        exit()

    m = data["m"]
    n = data["n"]
    task_data = data["tasks"]
    tasks = []

    # Input validation and task creation
    if n != len(task_data):
        print(
            f"Warning: Mismatch between n ({n}) and number of tasks found ({len(task_data)}). Using actual task count."
        )
        n = len(task_data)

    for key, val in task_data.items():
        if (
            not isinstance(val.get("work_amount"), (int, float))
            or val["work_amount"] < 0
        ):
            print(f"Error: Invalid work_amount for task {key}. Exiting.")
            exit()
        if (
            not isinstance(val.get("speed_up_factors"), list)
            or not val["speed_up_factors"]
        ):
            print(f"Error: Invalid speed_up_factors for task {key}. Exiting.")
            exit()

        speedups = val["speed_up_factors"]
        # Pad or truncate speedup factors to match 'm' processors (index 0 to m-1)
        if len(speedups) < m:
            last_val = speedups[-1] if speedups else 0
            speedups.extend([last_val] * (m - len(speedups)))
        elif len(speedups) > m:
            speedups = speedups[:m]
        # Ensure speedups are non-negative
        speedups = [max(0.0, float(s)) for s in speedups]

        task = Task(key, float(val["work_amount"]), speedups)
        tasks.append(task)

    print(f"Scheduling {n} tasks on {m} processors.")
    print("Initial tasks:")
    # for t in tasks: print(t) # Can be verbose
    print("-" * 20)

    makespan, schedule = greedy_scheduling(m=m, n=n, tasks=tasks)

    print(f"\nGreedy Algorithm Estimated Makespan: {makespan:.5f}")

    print("\nGenerated Schedule Log:")
    if not schedule:
        print("  (No schedule generated - check for errors or empty task list)")
    else:
        # Pretty print the schedule
        for entry in schedule:
            print(
                f"  Task {entry['task_id']}: "
                f"[{entry['start_time']:.4f} - {entry['end_time']:.4f}] "
                f"(Duration: {(entry['end_time'] - entry['start_time']):.4f}) "
                f"on {entry['num_processors']} processors"
            )

    # Example: Save schedule to a JSON file
    output_filename = "schedule_output_basic.json"

    try:
        with open(output_filename, "w") as f:
            json.dump(schedule, f, indent=2)
        logger.info(f"Schedule saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save schedule to {output_filename}: {e}")
