import heapq  # Can be useful for efficiently finding minimums
import json
import math
import random  # For distinct colors
from collections import defaultdict
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from msf.config.settings import settings
from msf.visualizer import Visualizer


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
