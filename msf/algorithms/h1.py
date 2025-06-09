import math
from enum import Enum

from loguru import logger

from msf.config.settings import settings
from msf.utils.models import FinalSchedule


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

    def __str__(self):
        return (
            f"CandidateAllocation(amount_of_work={self.amount_of_work:.5f}, "
            f"n_processors={self.n_processors}, "
            f"priority={self.priority}, "
            f"action={self.action.value}, "
            f"task_id={self.task.id})"
        )

    __repr__ = __str__


def get_candidate_allocations(
    waiting_tasks: dict[str, Task],
    running_tasks: dict[str, tuple[Task, int]],
    free_processors: int,
    max_processors: int,
) -> list[CandidateAllocation]:
    """
    Returns a list of candidate allocations for the waiting tasks and running tasks.
    For new tasks, considers optimal processor allocation based on speedup factors.
    The work amount is adjusted by the speedup factor to reflect actual processing time.
    """

    def get_best_allocation(
        task: Task, current_processors: int = 1
    ) -> tuple[int, float]:
        """Helper function to find the best allocation for a task"""
        # print(f"Checking task {task.id} with {current_processors} processors")

        best_processors = current_processors - 1
        best_speedup = task.speed_up_factors[best_processors]

        for processor_count in range(
            current_processors, current_processors + free_processors - 1
        ):
            current_speedup = task.speed_up_factors[processor_count]

            if current_speedup > best_speedup + settings.epsilon:
                best_speedup = current_speedup
                best_processors = processor_count

        # print(f"Best allocation for task {task.id}: {best_processors + 1} processors\n")

        return best_processors + 1, best_speedup

    candidates = []

    # WAITING TASKS
    for _, task in waiting_tasks.items():
        best_processors, best_speedup = get_best_allocation(task)

        if best_processors > free_processors:
            continue

        candidates.append(
            CandidateAllocation(
                amount_of_work=task.work_amount_left / best_speedup,
                n_processors=best_processors,
                priority=0,
                action=CandidateAction.START,
                task=task,
            )
        )

    # RUNNING TASKS
    for _, (task, current_p) in running_tasks.items():
        best_processors, best_speedup = get_best_allocation(task, current_p)
        if best_processors - current_p > free_processors:
            continue
        elif best_processors > current_p:
            candidates.append(
                CandidateAllocation(
                    amount_of_work=task.work_amount_left / best_speedup,
                    n_processors=best_processors,
                    priority=1,
                    action=CandidateAction.ADD,
                    task=task,
                )
            )

    return candidates


def h1_algorithm(m: int, n: int, tasks_dict: dict[str, dict]) -> FinalSchedule:
    """
    Greedy algorithm for malleable job scheduling (minimize makespan).
    Returns the makespan and a detailed schedule log.
    """

    tasks = {
        task_id: Task(
            id=task_id,
            work_amount=task["work_amount"],
            speed_up_factors=task["speed_up_factors"],
        )
        for task_id, task in tasks_dict.items()
    }
    current_time = 0.0
    free_processors = m
    waiting_tasks = {task.id: task for task in tasks.values()}
    running_tasks = {}
    completed_count = 0
    schedule_log = []
    active_intervals = {}

    while completed_count < n:
        # print(f"\nCreating interval \n -- free_processors: {free_processors}\n")
        while free_processors > 0:
            if not (
                candidates := get_candidate_allocations(
                    waiting_tasks, running_tasks, free_processors, max_processors=m
                )
            ):
                break

            candidates.sort(key=lambda x: (x.amount_of_work, x.priority, x.task.id))

            # print(f"Candidates: {candidates}")
            best_candidate = candidates[0]

            if best_candidate.action == CandidateAction.START:
                task_to_start = waiting_tasks.pop(best_candidate.task.id)
                running_tasks[task_to_start.id] = (
                    task_to_start,
                    best_candidate.n_processors,
                )
                free_processors -= best_candidate.n_processors
                active_intervals[task_to_start.id] = {
                    "start_time": current_time,
                    "num_processors": best_candidate.n_processors,
                }
            elif best_candidate.action == CandidateAction.ADD:
                task_to_add_to, p = running_tasks[best_candidate.task.id]
                additional_processors = best_candidate.n_processors - p

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
                running_tasks[best_candidate.task.id] = (
                    task_to_add_to,
                    best_candidate.n_processors,
                )
                free_processors -= additional_processors
            else:
                logger.error(f"Invalid allocation action: {best_candidate.action}")
                break

        # Calculate time to next completion
        min_delta_t = float("inf")
        tasks_to_complete = []

        for task_id, (task, p) in running_tasks.items():
            time_to_finish = task.work_amount_left / task.speed_up_factors[p - 1]

            if math.isclose(time_to_finish, min_delta_t, abs_tol=settings.epsilon):
                tasks_to_complete.append((time_to_finish, task_id))
            elif time_to_finish < min_delta_t:
                min_delta_t = time_to_finish
                tasks_to_complete = [(time_to_finish, task_id)]

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
                # If no tasks are running and we haven't completed all tasks,
                # we need to start new tasks
                if not running_tasks and waiting_tasks:
                    min_delta_t = 0
                else:
                    for task_id, (task, p) in running_tasks.items():
                        print(
                            f"Task {task_id} has {p} processors and {task.work_amount_left} work left"
                        )

                    for task_id, task in waiting_tasks.items():
                        print(f"Task {task_id} has {task.work_amount_left} work left")

                    print(f"Error: No task progress possible at time {current_time}")
                    break

        min_delta_t = max(0.0, min_delta_t)
        current_time += min_delta_t

        # Update work for running tasks
        if min_delta_t > 0:
            for task_id, (task, p) in running_tasks.items():
                work_done = task.speed_up_factors[p - 1] * min_delta_t
                task.work_amount_left -= work_done

        # First, check for any tasks that have completed
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

    if active_intervals:
        logger.error("Some tasks still have active intervals after completion")

    schedule_log.sort(key=lambda x: (x["start_time"], x["task_id"]))

    return FinalSchedule(
        makespan=current_time, schedule=schedule_log, num_processors=m, num_tasks=n
    )
