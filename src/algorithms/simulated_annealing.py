import copy
import math
import random
from enum import Enum
from typing import Any

from src.settings import settings

NEXT_SEGMENT_UID_COUNTER = 0


class Segment:
    """Represents a portion of a task's execution with a fixed processor allocation."""

    def __init__(self, task_id, segment_id, processors, work_amount_segment):
        self.task_id = task_id
        self.segment_id = segment_id
        self.processors = processors
        self.work_amount_segment = work_amount_segment
        self.duration = 0.0

    def __repr__(self):
        return (
            f"Segment(id={self.segment_id}, task={self.task_id}, "
            f"P={self.processors}, W={self.work_amount_segment:.2f})"
        )


class ScheduledSegmentOutput:
    """Represents a segment placed on the schedule, for final output."""

    def __init__(self, task_id, start_time, end_time, num_processors):
        self.task_id = task_id
        self.start_time = start_time
        self.end_time = end_time
        self.num_processors = num_processors

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "start_time": round(self.start_time, 4),  # Round for cleaner output
            "end_time": round(self.end_time, 4),
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
    """Returns a unique identifier for a segment."""
    global NEXT_SEGMENT_UID_COUNTER
    NEXT_SEGMENT_UID_COUNTER += 1
    return f"s{NEXT_SEGMENT_UID_COUNTER}"


# --- Instance Data Handling ---
def load_instance_data(instance_json):
    """Loads and pre-processes instance data."""
    n_tasks = instance_json["n"]
    m_processors = instance_json["m"]
    task_details_map = {}
    for task_id, details in instance_json["tasks"].items():
        # Ensure speed_up_factors array covers up to m_processors
        suf = list(details["speed_up_factors"])  # Make a mutable copy
        if len(suf) < m_processors:
            last_speedup = suf[-1] if suf else 1.0  # Fallback if empty
            suf.extend([last_speedup] * (m_processors - len(suf)))

        task_details_map[task_id] = {
            "total_work_amount": details["work_amount"],
            # 0-indexed: speed_up_factors[p-1] for p processors
            "speed_up_factors": suf[:m_processors],
            "max_allocatable_processors": len(
                details["speed_up_factors"]
            ),  # Original max for this task
        }
    return n_tasks, m_processors, task_details_map


def get_segment_duration(segment, task_info_map):
    """Calculates duration of a segment given its work and processor allocation."""
    task_id = segment.task_id
    speedup_idx = segment.processors - 1
    speedup = task_info_map[task_id]["speed_up_factors"][speedup_idx]

    return segment.work_amount_segment / speedup


def decode_solution(
    solution_task_segments_map: dict[str, list[Segment]],
    global_segment_order_ids: list[str],
    m_total_processors: int,
    task_info_map: dict[str, Any],
) -> tuple[list[ScheduledSegmentOutput], float]:
    """
    Constructs a schedule from the solution representation.

    Args:
        solution_task_segments_map: dict[str, list[Segment]] - Solution task segments map.
        global_segment_order_ids: list[str] - Global segment order IDs.
        m_total_processors: int - Total number of processors.
        task_info_map: dict[str, Any] - Task information map.

    Returns:
        tuple[list[ScheduledSegmentOutput], float] - Scheduled outputs and makespan.
    """
    scheduled_outputs = []
    makespan = 0.0

    # processor_finish_times[i] = time when physical system processor 'i' becomes free
    processor_finish_times = [0.0] * m_total_processors

    # Create a flat list of actual Segment objects in the specified global order
    segments_to_schedule = []
    for seg_id in global_segment_order_ids:
        found = False
        for task_id_key in solution_task_segments_map:
            for seg_obj in solution_task_segments_map[task_id_key]:
                if seg_obj.segment_id == seg_id:
                    segments_to_schedule.append(seg_obj)
                    found = True
                    break
            if found:
                break
        if not found:  # Should not happen if data is consistent
            # print(f"Error: Segment ID {seg_id} from global order not found in segment map.")
            return [], float("inf")

    for segment_to_place in segments_to_schedule:
        segment_to_place.duration = get_segment_duration(
            segment_to_place, task_info_map
        )

        if (
            segment_to_place.duration == float("inf")
            or segment_to_place.processors <= 0
            or segment_to_place.processors > m_total_processors
        ):
            return [], float("inf")  # Invalid segment implies infinitely bad schedule

        num_procs_needed = segment_to_place.processors

        earliest_start_time = 0.0
        assigned_proc_indices = []

        # Find the earliest time this segment can start (Greedy Approach)
        # Iterate through all unique processor finish times as potential start points
        # This ensures we check event points where processor availability might change.
        distinct_finish_times = sorted(list(set(processor_finish_times)))
        candidate_start_times = [0.0] + [
            t for t in distinct_finish_times if t > 1e-9
        ]  # Add 0 and positive finish times

        best_t_start_for_segment = float("inf")

        for t_candidate in candidate_start_times:
            available_procs_at_t_candidate = []
            for proc_idx in range(m_total_processors):
                if (
                    processor_finish_times[proc_idx] <= t_candidate + 1e-9
                ):  # Check if proc is free by t_candidate
                    available_procs_at_t_candidate.append(proc_idx)

            if len(available_procs_at_t_candidate) >= num_procs_needed:
                # Found enough processors. This t_candidate is a valid start time.
                best_t_start_for_segment = t_candidate
                assigned_proc_indices = available_procs_at_t_candidate[
                    :num_procs_needed
                ]
                break  # Found the earliest possible, due to sorted candidate_start_times

        if not assigned_proc_indices:  # Should only happen if num_procs_needed > m_total_processors (checked) or logic error
            # print(f"Decoder Error: Could not find slot for segment {segment_to_place}")
            return [], float("inf")

        earliest_start_time = best_t_start_for_segment
        segment_end_time = earliest_start_time + segment_to_place.duration

        # Update finish times for the assigned physical processors
        for proc_idx in assigned_proc_indices:
            processor_finish_times[proc_idx] = segment_end_time

        scheduled_outputs.append(
            ScheduledSegmentOutput(
                task_id=segment_to_place.task_id,
                start_time=earliest_start_time,
                end_time=segment_end_time,
                num_processors=num_procs_needed,
            )
        )
        makespan = max(makespan, segment_end_time)

    return scheduled_outputs, makespan


def generate_initial_solution(
    task_info_map: dict[str, Any],
    m_total_processors: int,
    initial_segments_per_task: int = 1,
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
            max_p_task_can_use = info["max_allocatable_processors"]
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

    # print(f"Current global order: {current_global_order}\n")
    # print(f"Current segments map: {current_segments_map}\n")
    # print(f"Possible moves: {possible_moves}\n")
    # print(f"Task info map: {task_info_map}\n")
    # print(f"M total processors: {m_total_processors}\n")

    # raise Exception("Stop here")

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

    def find_segment_and_task(segment_id_to_find, segments_map):
        for task_id, segment_list in segments_map.items():
            for segment in segment_list:
                if segment.segment_id == segment_id_to_find:
                    return task_id, segment
        return None, None

    match move_type:
        case NeighborMoveType.CHANGE_ALLOC:
            segment_id_to_modify = random.choice(new_global_order)
            task_id, segment_obj = find_segment_and_task(
                segment_id_to_modify, new_segments_map
            )
            if segment_obj:
                max_p_task = task_info_map[task_id]["max_allocatable_processors"]
                new_procs = segment_obj.processors
                # Try to ensure a change if multiple options exist
                if min(max_p_task, m_total_processors) > 1:
                    while (
                        new_procs == segment_obj.processors
                    ):  # Ensure it changes if possible
                        new_procs = random.randint(
                            1, min(max_p_task, m_total_processors)
                        )
                else:  # Only 1 processor option
                    new_procs = 1
                segment_obj.processors = new_procs

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
            if segment_to_split.work_amount_segment > settings.epsilon:
                original_work = segment_to_split.work_amount_segment
                split_ratio = random.uniform(0.25, 0.75)

                work1 = original_work * split_ratio
                work2 = original_work - work1

                max_p_task = task_info_map[task_id]["max_allocatable_processors"]
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
                max_p_task = task_info_map[task_to_merge_in][
                    "max_allocatable_processors"
                ]
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
        task_info_map, processor_num, initial_segments_per_task=None
    )
    current_schedule_list, current_makespan = decode_solution(
        current_segments_map, current_global_order, processor_num, task_info_map
    )

    # best_segments_map = copy.deepcopy(current_segments_map)
    # best_global_order = list(current_global_order)
    best_makespan = current_makespan
    best_schedule_list = list(current_schedule_list)

    temperature = initial_temperature
    total_iterations = 0

    print(f"Initial Makespan: {current_makespan:.4f}")

    while temperature > final_temperature:
        for i in range(iterations_per_temperature):
            total_iterations += 1
            neighbor_segments_map, neighbor_global_order = get_neighbor_solution(
                current_segments_map, current_global_order, task_info_map, processor_num
            )

            neighbor_schedule_list, neighbor_makespan = decode_solution(
                neighbor_segments_map,
                neighbor_global_order,
                processor_num,
                task_info_map,
            )

            if neighbor_makespan == float("inf"):
                continue

            delta_E = neighbor_makespan - current_makespan

            if delta_E < 0:
                current_segments_map = neighbor_segments_map
                current_global_order = neighbor_global_order
                current_makespan = neighbor_makespan
                current_schedule_list = neighbor_schedule_list

                if current_makespan < best_makespan:
                    # best_segments_map = copy.deepcopy(current_segments_map)
                    # best_global_order = list(current_global_order)
                    best_makespan = current_makespan
                    best_schedule_list = list(current_schedule_list)
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
    return [s.to_dict() for s in best_schedule_list], best_makespan
