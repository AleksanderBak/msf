from typing import Any

from msf.algorithms.simulated_annealing import (
    FinalSchedule,
    Segment,
    load_instance_data,
    simulated_annealing,
)
from msf.algorithms.simulated_annealing import (
    decode_solution as decode_solution_original,
)
from msf.utils.helpers import group_intervals, read_instance_data
from msf.utils.test_data_lalala import segment_order, segments_map
from msf.utils.validator import validate_schedule
from msf.visualizer import Visualizer


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


def get_segment_duration(
    task_id: str, work_amount: float, task_info_map: dict[str, Any], processors: int
) -> float:
    speedup = task_info_map[task_id]["speed_up_factors"][processors - 1]

    return work_amount / speedup


class SegmentScheduler:
    def __init__(self, total_processors: int, task_info_map: dict[str, Any]):
        self.total_processors = total_processors
        self.schedule = []
        self.current_time = 0
        self.task_info_map = task_info_map

    def schedule_segments(
        self,
        current_interval_start: float,
        running_tasks: dict[str, Any],
        free_processors: int,
    ) -> tuple[dict[str, Any], int, float]:
        print(
            f"\n=========Scheduling segments at time {current_interval_start}========"
        )
        tasks_to_end = []
        current_interval_end = min(task["end_time"] for task in running_tasks.values())
        print(f"Current interval end: {current_interval_end}\n")

        for task_id, task_info in running_tasks.items():
            self.schedule.append(
                ScheduledSegmentOutput(
                    task_id=task_id,
                    start_time=current_interval_start,
                    end_time=current_interval_end,
                    num_processors=task_info["processors_to_use"],
                )
            )
            if task_info["end_time"] == current_interval_end:
                print(f"Ending task {task_id}")
                tasks_to_end.append(task_id)
                free_processors += task_info["processors_to_use"]

        for task_id in tasks_to_end:
            running_tasks.pop(task_id)

        new_interval_start = current_interval_end
        return running_tasks, free_processors, new_interval_start

    def decode_solution(
        self,
        segment_map: dict[str, list[Segment]],
        segment_order: list[str],
        task_info_map: dict[str, Any],
        processors: int,
    ) -> tuple[list[ScheduledSegmentOutput], float]:
        segments = {}
        segments_to_schedule = segment_order.copy()

        for task_segments in segment_map.values():
            for seg in task_segments:
                segments[seg.segment_id] = seg

        current_interval_start = 0
        running_tasks = {}
        free_processors = processors

        while segments_to_schedule:
            if free_processors == 0:
                print(f"Running tasks when processors are 0: {running_tasks}")
                running_tasks, free_processors, current_interval_start = (
                    self.schedule_segments(
                        current_interval_start, running_tasks, free_processors
                    )
                )
                print(f"Current running tasks after scheduling: {running_tasks}")

            segment_candidate = None

            for seg_id in segments_to_schedule:
                task_id = segments[seg_id].task_id
                if task_id not in running_tasks:
                    if segments[seg_id].processors <= free_processors:
                        segment_candidate = seg_id
                        break
                else:
                    if (
                        running_tasks[task_id]["processors_to_use"]
                        + segments[seg_id].processors
                        <= free_processors
                    ):
                        segment_candidate = seg_id
                        break

            if segment_candidate is None:
                print("No segment candidate found")
                segment_candidate = segments_to_schedule[0]
                print(f"Segment candidate: {segment_candidate}")

            seg_id = segment_candidate
            if seg_id is not None:
                segments_to_schedule.remove(seg_id)
                task_id = segments[seg_id].task_id
                max_proc_number = min(
                    free_processors,
                    task_info_map[task_id]["max_processors"],
                )

                if task_id in running_tasks:
                    print(f"Task {task_id} is already running")

                    new_processors_to_use = min(
                        max_proc_number, segments[seg_id].processors
                    )
                    processors_to_use = (
                        running_tasks[task_id]["processors_to_use"]
                        + new_processors_to_use
                    )

                    current_work_amount = (
                        running_tasks[task_id]["end_time"] - current_interval_start
                    ) * task_info_map[task_id]["speed_up_factors"][
                        running_tasks[task_id]["processors_to_use"] - 1
                    ]
                    new_work_amount = (
                        current_work_amount + segments[seg_id].work_amount_segment
                    )
                    # print(f"Current work amount: {current_work_amount}")
                    # print(f"New work amount: {new_work_amount}")
                    running_tasks[task_id]["processors_to_use"] = processors_to_use
                    duration = get_segment_duration(
                        task_id,
                        new_work_amount,
                        task_info_map,
                        processors_to_use,
                    )
                    print(
                        f"Current task {task_id} end time: {running_tasks[task_id]['end_time']}\n"
                        f"Current interval start: {current_interval_start}\n"
                        f"Duration: {duration}\n"
                        f"New end time: {current_interval_start + duration}\n"
                    )
                    running_tasks[task_id]["end_time"] = (
                        current_interval_start + duration
                    )
                else:
                    new_processors_to_use = min(
                        max_proc_number, segments[seg_id].processors
                    )
                    print(f"Adding segment of new task {task_id}")
                    duration = get_segment_duration(
                        task_id,
                        segments[seg_id].work_amount_segment,
                        task_info_map,
                        new_processors_to_use,
                    )
                    running_tasks[task_id] = {
                        "end_time": current_interval_start + duration,
                        "processors_to_use": new_processors_to_use,
                    }
                    processors_to_use = new_processors_to_use

                print(f"Using {processors_to_use} processors for segment {seg_id}")
                free_processors -= new_processors_to_use
            else:
                print("No segment candidate found")
                free_processors = 0
                continue

        print("========All segments scheduled========")
        print(f"Running tasks: {running_tasks}")
        while running_tasks:
            print(f"Currently available processors: {free_processors}")
            print(f"Current interval start: {current_interval_start}")

            while free_processors > 0:
                task_to_expand = None

                for task_id, task_info in running_tasks.items():
                    if (
                        task_info["processors_to_use"]
                        < task_info_map[task_id]["max_processors"]
                    ):
                        task_to_expand = task_id
                        break

                if task_to_expand is not None:
                    processors_to_add = min(
                        task_info_map[task_to_expand]["max_processors"]
                        - running_tasks[task_to_expand]["processors_to_use"],
                        free_processors,
                    )
                    print(
                        f"Adding {processors_to_add} processors to task {task_to_expand} with limit: {task_info_map[task_to_expand]['max_processors']}"
                    )
                    new_processors_to_use = (
                        running_tasks[task_to_expand]["processors_to_use"]
                        + processors_to_add
                    )

                    work_amount = (
                        running_tasks[task_to_expand]["end_time"]
                        - current_interval_start
                    ) * task_info_map[task_to_expand]["speed_up_factors"][
                        running_tasks[task_to_expand]["processors_to_use"] - 1
                    ]

                    new_duration = get_segment_duration(
                        task_to_expand,
                        work_amount,
                        task_info_map,
                        new_processors_to_use,
                    )
                    running_tasks[task_to_expand]["end_time"] = (
                        current_interval_start + new_duration
                    )
                    running_tasks[task_to_expand]["processors_to_use"] = (
                        new_processors_to_use
                    )
                    free_processors -= processors_to_add
                    print(f"Expanding task {task_to_expand}")
                else:
                    break

            print(f"Running tasks: {running_tasks}")
            running_tasks, free_processors, current_interval_start = (
                self.schedule_segments(
                    current_interval_start, running_tasks, free_processors
                )
            )

        return self.schedule, current_interval_start


if __name__ == "__main__":
    # instance_data = {
    #     "m": 4,
    #     "n": 3,
    #     "tasks": {
    #         "T1": {
    #             "work_amount": 40,
    #             "speed_up_factors": [1.0, 2.0, 3.0, 4.0],
    #         },
    #         "T2": {
    #             "work_amount": 20,
    #             "speed_up_factors": [1.0, 2.0, 3.0, 4.0],
    #         },
    #         "T3": {
    #             "work_amount": 5,
    #             "speed_up_factors": [1.0, 2.0, 3.0, 4.0],
    #         },
    #     },
    # }

    # segments_map = {
    #     "T1": [
    #         Segment(
    #             task_id="T1", segment_id="S1", processors=2, work_amount_segment=10
    #         ),
    #         Segment(
    #             task_id="T1", segment_id="S2", processors=1, work_amount_segment=10
    #         ),
    #         Segment(
    #             task_id="T1", segment_id="S3", processors=1, work_amount_segment=20
    #         ),
    #     ],
    #     "T2": [
    #         Segment(task_id="T2", segment_id="S4", processors=3, work_amount_segment=20)
    #     ],
    #     "T3": [
    #         Segment(task_id="T3", segment_id="S5", processors=1, work_amount_segment=5)
    #     ],
    # }
    # segment_order = ["S1", "S4", "S2", "S5", "S3"]

    instance_data = read_instance_data("instance_data/instance_1.json")

    n_tasks, m_processors, task_info_map = load_instance_data(instance_data)

    segments_map, segment_order, _ = simulated_annealing(
        instance_data or {},
        initial_temperature=50,
        final_temperature=5,
        cooling_rate=0.98,
        iterations_per_temperature=50,
        initial_segments_per_task=2,
    )

    segment_scheduler = SegmentScheduler(m_processors, task_info_map)

    schedule, makespan = segment_scheduler.decode_solution(
        segments_map, segment_order, task_info_map, m_processors
    )

    if schedule and makespan:
        interval_list = [s.to_dict() for s in schedule]
        interval_list_group = group_intervals(interval_list)
        # print(f"Makespan: {makespan}")
        # for i in interval_list:
        #     print(i)

        final_schedule = FinalSchedule(
            schedule=interval_list_group,
            makespan=makespan,
            num_processors=m_processors,
            num_tasks=n_tasks,
        )

        res = validate_schedule(final_schedule, instance_data or {})
        if res:
            print("Schedule is valid")
        else:
            print("Schedule is invalid")

        visualizer = Visualizer()
        visualizer.visualize_schedule_by_processor(
            final_schedule,
            save_to_file=True,
            filename="test.png",
        )
