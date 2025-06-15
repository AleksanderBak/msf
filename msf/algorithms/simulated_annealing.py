import copy
import math
import random
from collections import deque
from enum import Enum
from typing import Any

from loguru import logger

from msf.config.settings import settings
from msf.utils.helpers import FinalSchedule, group_intervals

NEXT_SEGMENT_UID_COUNTER = 0


class Segment:
    """Represents a portion of a task's execution with a fixed processor allocation."""

    def __init__(
        self, task_id: str, segment_id: str, processors: int, work_amount_segment: float
    ):
        self.task_id = task_id
        self.segment_id = segment_id
        self.processors = processors
        self.work_amount_segment = work_amount_segment
        self.duration = 0.0

    def __repr__(self):
        return (
            f"Segment(segment_id={self.segment_id}, task_id={self.task_id}, "
            f"processors={self.processors}, work_amount_segment={self.work_amount_segment})"
        )


class ScheduledSegmentOutput:
    """Represents a segment placed on the schedule, for final output."""

    def __init__(
        self, task_id: str, start_time: float, end_time: float, num_processors: int
    ):
        self.task_id = task_id
        self.start_time = start_time
        self.end_time = end_time
        self.num_processors = num_processors

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_processors": self.num_processors,
        }


class NeighborMoveType(str, Enum):
    """Represents the type of neighbor move to be made."""

    CHANGE_ALLOC = "change_alloc"
    CHANGE_ORDER_SWAP = "change_order_swap"
    CHANGE_ORDER_MOVE = "change_order_move"
    SPLIT_SEGMENT = "split_segment"
    MERGE_SEGMENTS = "merge_segments"
    SHIFT_WORK = "shift_work"


def get_next_segment_uid() -> str:
    global NEXT_SEGMENT_UID_COUNTER
    NEXT_SEGMENT_UID_COUNTER += 1
    return f"s{NEXT_SEGMENT_UID_COUNTER}"


def load_instance_data(
    instance_json: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    n_tasks = instance_json["n"]
    m_processors = instance_json["m"]
    task_details_map = {}
    for task_id, details in instance_json["tasks"].items():
        best_speedup = 1.0
        best_speedup_idx = 0
        for idx, speedup_factor in enumerate(details["speed_up_factors"]):
            if speedup_factor > best_speedup:
                best_speedup = speedup_factor
                best_speedup_idx = idx

        task_details_map[task_id] = {
            "total_work_amount": details["work_amount"],
            "speed_up_factors": details["speed_up_factors"],
            "max_processors": best_speedup_idx + 1,
        }
    return n_tasks, m_processors, task_details_map


def get_segment_duration(
    task_id: str, work_amount: float, task_info_map: dict[str, Any], processors: int
) -> float:
    speedup = task_info_map[task_id]["speed_up_factors"][processors - 1]

    return work_amount / speedup


class SegmentScheduler:
    def __init__(self, total_processors: int, task_info_map: dict[str, Any]):
        self.total_processors = total_processors
        self.task_info_map = task_info_map

    def _flush_running_tasks(
        self,
        interval_start: float,
        running_tasks: dict[str, Any],
        available_processors: int,
        schedule: list[ScheduledSegmentOutput],
    ) -> tuple[dict[str, Any], int, float]:
        tasks_to_finish = []
        interval_end = min(task["end_time"] for task in running_tasks.values())

        for task_id, task_info in running_tasks.items():
            schedule.append(
                ScheduledSegmentOutput(
                    task_id=task_id,
                    start_time=interval_start,
                    end_time=interval_end,
                    num_processors=task_info["processors_to_use"],
                )
            )
            if task_info["end_time"] == interval_end:
                tasks_to_finish.append(task_id)
                available_processors += task_info["processors_to_use"]

        for task_id in tasks_to_finish:
            running_tasks.pop(task_id)

        new_interval_start = interval_end
        return running_tasks, available_processors, new_interval_start

    def _find_next_segment(
        self, segments, segments_to_schedule, running_tasks, available_processors
    ) -> str:
        def can_schedule_with_constraints(seg_id: str) -> bool:
            segment = segments[seg_id]
            task_id = segment.task_id

            if task_id not in running_tasks:
                return segment.processors <= available_processors

            return (
                running_tasks[task_id]["processors_to_use"] + segment.processors
                <= available_processors
            )

        def can_schedule_without_constraints(seg_id: str) -> bool:
            segment = segments[seg_id]
            task_id = segment.task_id

            if task_id not in running_tasks:
                return True

            return (
                running_tasks[task_id]["processors_to_use"]
                < self.task_info_map[task_id]["max_processors"]
            )

        for seg_id in segments_to_schedule:
            if can_schedule_with_constraints(seg_id):
                return seg_id

        for seg_id in segments_to_schedule:
            if can_schedule_without_constraints(seg_id):
                return seg_id

        logger.warning(
            f"No possible segment found for {available_processors} processors, returning first element on the list"
        )
        return segments_to_schedule[0]

    def _build_segments_dict(
        self, segment_map: dict[str, list[Segment]]
    ) -> dict[str, Segment]:
        segments = {}
        for task_segments in segment_map.values():
            for seg in task_segments:
                segments[seg.segment_id] = seg
        return segments

    def decode_solution(
        self,
        segment_map: dict[str, list[Segment]],
        segment_order: list[str],
        processors: int,
    ) -> tuple[list[ScheduledSegmentOutput], float]:
        schedule = []
        segments = self._build_segments_dict(segment_map)
        segments_to_schedule = segment_order.copy()

        running_tasks = {}
        free_processors = processors
        current_interval_start = 0

        # print(f"Segments to schedule: \n{segments_to_schedule}")
        # print(f"Segments map: \n{segments}")

        # Main scheduling loop
        while segments_to_schedule:
            if free_processors == 0:
                running_tasks, free_processors, current_interval_start = (
                    self._flush_running_tasks(
                        current_interval_start, running_tasks, free_processors, schedule
                    )
                )

            next_segment_id = self._find_next_segment(
                segments, segments_to_schedule, running_tasks, free_processors
            )

            if next_segment_id is not None:
                segments_to_schedule.remove(next_segment_id)
                task_id = segments[next_segment_id].task_id
                max_proc_number = min(
                    free_processors,
                    self.task_info_map[task_id]["max_processors"],
                )

                if task_id in running_tasks:
                    new_processors_to_use = (
                        max_proc_number - running_tasks[task_id]["processors_to_use"]
                    )
                    processors_to_use = (
                        running_tasks[task_id]["processors_to_use"]
                        + new_processors_to_use
                    )
                    current_work_amount = (
                        running_tasks[task_id]["end_time"] - current_interval_start
                    ) * self.task_info_map[task_id]["speed_up_factors"][
                        running_tasks[task_id]["processors_to_use"] - 1
                    ]
                    new_work_amount = (
                        current_work_amount
                        + segments[next_segment_id].work_amount_segment
                    )

                    running_tasks[task_id]["processors_to_use"] = processors_to_use
                    duration = get_segment_duration(
                        task_id,
                        new_work_amount,
                        self.task_info_map,
                        processors_to_use,
                    )
                    running_tasks[task_id]["end_time"] = (
                        current_interval_start + duration
                    )
                else:
                    new_processors_to_use = min(
                        max_proc_number, segments[next_segment_id].processors
                    )
                    duration = get_segment_duration(
                        task_id,
                        segments[next_segment_id].work_amount_segment,
                        self.task_info_map,
                        new_processors_to_use,
                    )
                    running_tasks[task_id] = {
                        "end_time": current_interval_start + duration,
                        "processors_to_use": new_processors_to_use,
                    }
                    processors_to_use = new_processors_to_use

                free_processors -= new_processors_to_use
            else:
                free_processors = 0
                continue
        # Finish started segments
        while running_tasks:
            while free_processors > 0:
                task_to_expand = None

                for task_id, task_info in running_tasks.items():
                    if (
                        task_info["processors_to_use"]
                        < self.task_info_map[task_id]["max_processors"]
                    ):
                        task_to_expand = task_id
                        break

                if task_to_expand is not None:
                    processors_to_add = min(
                        self.task_info_map[task_to_expand]["max_processors"]
                        - running_tasks[task_to_expand]["processors_to_use"],
                        free_processors,
                    )
                    new_processors_to_use = (
                        running_tasks[task_to_expand]["processors_to_use"]
                        + processors_to_add
                    )

                    work_amount = (
                        running_tasks[task_to_expand]["end_time"]
                        - current_interval_start
                    ) * self.task_info_map[task_to_expand]["speed_up_factors"][
                        running_tasks[task_to_expand]["processors_to_use"] - 1
                    ]

                    new_duration = get_segment_duration(
                        task_to_expand,
                        work_amount,
                        self.task_info_map,
                        new_processors_to_use,
                    )
                    running_tasks[task_to_expand]["end_time"] = (
                        current_interval_start + new_duration
                    )
                    running_tasks[task_to_expand]["processors_to_use"] = (
                        new_processors_to_use
                    )
                    free_processors -= processors_to_add
                else:
                    break
            running_tasks, free_processors, current_interval_start = (
                self._flush_running_tasks(
                    current_interval_start, running_tasks, free_processors, schedule
                )
            )

        return schedule, current_interval_start


def generate_initial_solution(
    task_info_map: dict[str, Any],
    m_total_processors: int,
    initial_segments_per_task: int | None,
) -> tuple[dict[str, list[Segment]], list[str]]:
    """
    Generates an initial solution for the simulated annealing algorithm.

    Args:
        task_info_map: dict[str, Any] - Task information map.
        m_total_processors: int - Total number of processors.
        initial_segments_per_task: int - Number of initial segments per task.

    Returns:
        tuple[dict[str, list[Segment]], list[str]] - Initial solution.
    """
    global NEXT_SEGMENT_UID_COUNTER
    NEXT_SEGMENT_UID_COUNTER = 0

    solution_task_segments_map = {task_id: [] for task_id in task_info_map}
    global_segment_order_ids = []

    for task_id, info in task_info_map.items():
        if initial_segments_per_task is None:
            num_initial_segments = random.randint(1, 2)
        else:
            num_initial_segments = initial_segments_per_task

        work_per_segment = info["total_work_amount"] / num_initial_segments

        for _ in range(num_initial_segments):
            seg_id = get_next_segment_uid()
            # Ensure initial proc alloc is valid for the task and system
            max_p_task_can_use = info["max_processors"]
            proc_alloc = random.randint(1, min(max_p_task_can_use, m_total_processors))

            segment = Segment(task_id, seg_id, proc_alloc, work_per_segment)
            solution_task_segments_map[task_id].append(segment)
            global_segment_order_ids.append(seg_id)

    random.shuffle(global_segment_order_ids)
    return solution_task_segments_map, global_segment_order_ids


def get_neighbor_solution(
    current_segments_map: dict[str, list[Segment]],
    current_global_order: list[str],
    task_info_map: dict[str, Any],
    m_total_processors: int,
) -> tuple[dict[str, list[Segment]], list[str]]:
    """
    Generates a neighbor solution for the simulated annealing algorithm.

    Args:
        current_segments_map: dict[str, list[Segment]] - Current segments map.
        current_global_order: list[str] - Current global order.
        task_info_map: dict[str, Any] - Task information map.
        m_total_processors: int - Total number of processors.

    Returns:
        tuple[dict[str, list[Segment]], list[str]] - Neighbor solution.
    """
    new_segments_map = copy.deepcopy(current_segments_map)
    new_global_order = list(current_global_order)

    possible_moves = [
        NeighborMoveType.CHANGE_ALLOC,
        NeighborMoveType.CHANGE_ORDER_SWAP,
    ]
    if len(new_global_order) > 1:
        possible_moves.append(NeighborMoveType.CHANGE_ORDER_MOVE)

    has_segments = False
    has_multiple_segments = False

    for segment_list in new_segments_map.values():
        if len(segment_list) > 0:
            has_segments = True
        if len(segment_list) >= 2:
            has_multiple_segments = True
            break

    if has_segments:
        possible_moves.append(NeighborMoveType.SPLIT_SEGMENT)

    if has_multiple_segments:
        possible_moves.extend(
            [NeighborMoveType.SHIFT_WORK, NeighborMoveType.MERGE_SEGMENTS]
        )

    move_type = random.choice(possible_moves)

    def find_segment_and_task(
        segment_id_to_find: str, segments_map: dict[str, list[Segment]]
    ) -> tuple[str, Segment] | tuple[None, None]:
        for task_id, segment_list in segments_map.items():
            for segment in segment_list:
                if segment.segment_id == segment_id_to_find:
                    return task_id, segment
        return None, None

    match move_type:
        case NeighborMoveType.CHANGE_ALLOC:
            segment_id = random.choice(new_global_order)
            task_id, segment = find_segment_and_task(segment_id, new_segments_map)
            if segment:
                max_p_task = task_info_map[task_id]["max_processors"]
                new_procs = segment.processors

                # Try to ensure a change if multiple options exist
                if min(max_p_task, m_total_processors) > 1:
                    while (
                        new_procs == segment.processors
                    ):  # Ensure it changes if possible
                        new_procs = random.randint(
                            1, min(max_p_task, m_total_processors)
                        )
                else:  # Only 1 processor option
                    new_procs = 1
                segment.processors = new_procs

        case NeighborMoveType.CHANGE_ORDER_SWAP if len(new_global_order) >= 2:
            idx1, idx2 = random.sample(range(len(new_global_order)), 2)
            new_global_order[idx1], new_global_order[idx2] = (
                new_global_order[idx2],
                new_global_order[idx1],
            )

        case NeighborMoveType.CHANGE_ORDER_MOVE if len(new_global_order) >= 2:
            segment_to_move_idx = random.randrange(len(new_global_order))
            seg_id_moved = new_global_order.pop(segment_to_move_idx)
            new_pos = random.randrange(len(new_global_order) + 1)
            new_global_order.insert(new_pos, seg_id_moved)

        case NeighborMoveType.SPLIT_SEGMENT:
            seg_id_to_split = random.choice(new_global_order)
            task_id, segment_to_split = find_segment_and_task(
                seg_id_to_split, new_segments_map
            )
            if (
                segment_to_split
                and segment_to_split.work_amount_segment > settings.epsilon
            ):
                original_work = segment_to_split.work_amount_segment
                split_ratio = random.uniform(0.25, 0.75)

                work1 = original_work * split_ratio
                work2 = original_work - work1

                max_p_task = task_info_map[task_id]["max_processors"]
                procs1 = random.randint(1, min(max_p_task, m_total_processors))
                procs2 = random.randint(1, min(max_p_task, m_total_processors))

                new_seg_id1 = get_next_segment_uid()
                new_seg_id2 = get_next_segment_uid()
                new_seg1 = Segment(task_id, new_seg_id1, procs1, work1)
                new_seg2 = Segment(task_id, new_seg_id2, procs2, work2)

                new_segments_map[task_id] = [
                    s
                    for s in new_segments_map[task_id]
                    if s.segment_id != seg_id_to_split
                ]
                new_segments_map[task_id].extend([new_seg1, new_seg2])

                idx_in_global = new_global_order.index(seg_id_to_split)
                new_global_order.pop(idx_in_global)

                new_global_order.insert(idx_in_global, new_seg_id1)
                new_global_order.insert(idx_in_global + 1, new_seg_id2)

        case NeighborMoveType.MERGE_SEGMENTS:
            tasks_with_multiple_segments = [
                task_id
                for task_id, segment_list in new_segments_map.items()
                if len(segment_list) >= 2
            ]
            if tasks_with_multiple_segments:
                task_to_merge_in = random.choice(tasks_with_multiple_segments)

                segment_1, segment_2 = random.sample(
                    new_segments_map[task_to_merge_in], 2
                )

                merged_work = (
                    segment_1.work_amount_segment + segment_2.work_amount_segment
                )
                max_p_task = task_info_map[task_to_merge_in]["max_processors"]
                merged_procs = random.randint(1, min(max_p_task, m_total_processors))

                merged_segment_id = get_next_segment_uid()
                merged_segment = Segment(
                    task_to_merge_in, merged_segment_id, merged_procs, merged_work
                )

                new_segments_map[task_to_merge_in] = [
                    s
                    for s in new_segments_map[task_to_merge_in]
                    if s.segment_id not in [segment_1.segment_id, segment_2.segment_id]
                ]
                new_segments_map[task_to_merge_in].append(merged_segment)

                indices_in_global = []
                if segment_1.segment_id in new_global_order:
                    indices_in_global.append(
                        new_global_order.index(segment_1.segment_id)
                    )
                if segment_2.segment_id in new_global_order:
                    indices_in_global.append(
                        new_global_order.index(segment_2.segment_id)
                    )

                new_global_order = [
                    s_id
                    for s_id in new_global_order
                    if s_id not in [segment_1.segment_id, segment_2.segment_id]
                ]
                insert_pos = min(indices_in_global) if indices_in_global else 0
                new_global_order.insert(insert_pos, merged_segment_id)

        case NeighborMoveType.SHIFT_WORK:
            tasks_with_multiple_segments = [
                task_id
                for task_id, segment_list in new_segments_map.items()
                if len(segment_list) >= 2
            ]
            if tasks_with_multiple_segments:
                choosen_task = random.choice(tasks_with_multiple_segments)
                segment_1, segment_2 = random.sample(new_segments_map[choosen_task], 2)

                if segment_1.work_amount_segment > settings.epsilon:
                    shift_amount = (
                        random.uniform(0.1, 0.5) * segment_1.work_amount_segment
                    )  # Shift 10-50%

                    if segment_1.work_amount_segment - shift_amount > settings.epsilon:
                        segment_1.work_amount_segment -= shift_amount
                        segment_2.work_amount_segment += shift_amount

    return new_segments_map, new_global_order


def simulated_annealing(
    instance_data: dict[str, Any],
    initial_temperature: float,
    final_temperature: float,
    cooling_rate: float,
    iterations_per_temperature: int,
    non_improving_acceptance_factor: float = 1.0,
    initial_segments_per_task: int = 1,
    show_progress: bool = False,
):
    """
    Simulated Annealing Algorithm for malleable task scheduling to minimize makespan.

    Args:
        instance_data: dict[str, Any] - Instance data containing task information.
        initial_temperature: float - Initial temperature for the annealing process.
        final_temperature: float - Final temperature for the annealing process.
        cooling_rate: float - Cooling rate for the annealing process.
        iterations_per_temperature: int - Number of iterations per temperature.
        non_improving_acceptance_factor: float - Factor for non-improving moves.
        initial_segments_per_task: int | None - Number of initial segments per task.

    Returns:
        list[dict[str, Any]] - List of scheduled segments.
        float - Final makespan.
    """
    n_tasks, processor_num, task_info_map = load_instance_data(instance_data)

    current_segments_map, current_global_order = generate_initial_solution(
        task_info_map,
        processor_num,
        initial_segments_per_task=initial_segments_per_task,
    )

    segment_scheduler = SegmentScheduler(processor_num, task_info_map)

    current_schedule_list, current_makespan = segment_scheduler.decode_solution(
        current_segments_map, current_global_order, processor_num
    )

    best_makespan = current_makespan
    best_segments_map = current_segments_map
    best_global_order = current_global_order
    best_schedule_list = current_schedule_list

    temperature = initial_temperature
    total_iterations = 0

    print(f"Initial Makespan: {current_makespan:.4f}")

    while temperature > final_temperature:
        for i in range(iterations_per_temperature):
            total_iterations += 1
            neighbor_segments_map, neighbor_global_order = get_neighbor_solution(
                current_segments_map, current_global_order, task_info_map, processor_num
            )

            neighbor_schedule_list, neighbor_makespan = (
                segment_scheduler.decode_solution(
                    neighbor_segments_map,
                    neighbor_global_order,
                    processor_num,
                )
            )

            delta_E = neighbor_makespan - current_makespan

            if delta_E < 0:
                current_segments_map = neighbor_segments_map
                current_global_order = neighbor_global_order
                current_makespan = neighbor_makespan
                current_schedule_list = neighbor_schedule_list

                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_segments_map = current_segments_map
                    best_global_order = current_global_order
                    best_schedule_list = current_schedule_list
            else:
                boltzman_prob = math.exp(-delta_E / temperature)

                acceptance_probability = boltzman_prob * non_improving_acceptance_factor

                if random.random() < acceptance_probability:
                    current_segments_map = neighbor_segments_map
                    current_global_order = neighbor_global_order
                    current_makespan = neighbor_makespan
                    current_schedule_list = neighbor_schedule_list

        temperature *= cooling_rate
        if show_progress:
            if total_iterations % (iterations_per_temperature * 5) == 0:
                print(
                    f"Iter {total_iterations}, T={temperature:.2f}, Best makespan={best_makespan:.4f}"
                )

    print(f"\nFinished SA. Total iterations: {total_iterations}")

    interval_list = [s.to_dict() for s in best_schedule_list]
    interval_list_group = group_intervals(interval_list)

    return FinalSchedule(
        schedule=interval_list_group,
        makespan=best_makespan,
        num_processors=processor_num,
        num_tasks=n_tasks,
    )
