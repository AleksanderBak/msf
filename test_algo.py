from typing import Any

from loguru import logger

from msf.algorithms.simulated_annealing import (
    FinalSchedule,
    Segment,
    load_instance_data,
    simulated_annealing,
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

    def _flush_running_tasks(
        self,
        interval_start: float,
        running_tasks: dict[str, Any],
        available_processors: int,
    ) -> tuple[dict[str, Any], int, float]:
        tasks_to_finish = []
        interval_end = min(task["end_time"] for task in running_tasks.values())

        for task_id, task_info in running_tasks.items():
            self.schedule.append(
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
                < task_info_map[task_id]["max_processors"]
            )

        for seg_id in segments_to_schedule:
            if can_schedule_with_constraints(seg_id):
                return seg_id

        for seg_id in segments_to_schedule:
            if can_schedule_without_constraints(seg_id):
                return seg_id

        logger.warning("No possible segment found, returning first element on the list")
        return segments_to_schedule[0]

    def _build_segments_dict(
        self, segment_map: dict[str, list[Segment]]
    ) -> dict[str, Segment]:
        segments = {}
        for task_segments in segment_map.values():
            for seg in task_segments:
                segments[seg.segment_id] = seg
        return segments

    def _schedule_all_segments(
        self,
        segments: dict[str, Segment],
        segments_to_schedule: list[str],
        running_tasks: dict[str, Any],
        free_processors: int,
        current_interval_start: float,
        task_info_map: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        while segments_to_schedule:
            if free_processors == 0:
                running_tasks, free_processors, current_interval_start = (
                    self._flush_running_tasks(
                        current_interval_start, running_tasks, free_processors
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
                    task_info_map[task_id]["max_processors"],
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
                    ) * task_info_map[task_id]["speed_up_factors"][
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
                        task_info_map,
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
                        task_info_map,
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

        return running_tasks, current_interval_start

    def decode_solution(
        self,
        segment_map: dict[str, list[Segment]],
        segment_order: list[str],
        task_info_map: dict[str, Any],
        processors: int,
    ) -> tuple[list[ScheduledSegmentOutput], float]:
        segments = self._build_segments_dict(segment_map)
        segments_to_schedule = segment_order.copy()

        running_tasks = {}
        free_processors = processors
        current_interval_start = 0

        # Main scheduling loop
        while segments_to_schedule:
            if free_processors == 0:
                running_tasks, free_processors, current_interval_start = (
                    self._flush_running_tasks(
                        current_interval_start, running_tasks, free_processors
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
                    task_info_map[task_id]["max_processors"],
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
                    ) * task_info_map[task_id]["speed_up_factors"][
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
                        task_info_map,
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
                        task_info_map,
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
                else:
                    break
            running_tasks, free_processors, current_interval_start = (
                self._flush_running_tasks(
                    current_interval_start, running_tasks, free_processors
                )
            )

        return self.schedule, current_interval_start


if __name__ == "__main__":
    instance_data = read_instance_data("instance_data/instance_1.json")

    n_tasks, m_processors, task_info_map = load_instance_data(instance_data)

    # SA

    segments_map = {
        "T1": [
            Segment(
                segment_id="s15", task_id="T1", processors=4, work_amount_segment=100.0
            )
        ],
        "T2": [
            Segment(
                segment_id="s4", task_id="T2", processors=2, work_amount_segment=40.0
            ),
            Segment(
                segment_id="s11",
                task_id="T2",
                processors=1,
                work_amount_segment=13.918410806720898,
            ),
            Segment(
                segment_id="s12",
                task_id="T2",
                processors=2,
                work_amount_segment=26.0815891932791,
            ),
        ],
        "T3": [
            Segment(
                segment_id="s6",
                task_id="T3",
                processors=2,
                work_amount_segment=81.4564600142864,
            ),
            Segment(
                segment_id="s17",
                task_id="T3",
                processors=2,
                work_amount_segment=21.100696346820598,
            ),
            Segment(
                segment_id="s18",
                task_id="T3",
                processors=2,
                work_amount_segment=17.442843638892995,
            ),
        ],
        "T4": [
            Segment(
                segment_id="s7", task_id="T4", processors=1, work_amount_segment=25.0
            ),
            Segment(
                segment_id="s8", task_id="T4", processors=2, work_amount_segment=25.0
            ),
        ],
        "T5": [
            Segment(
                segment_id="s14",
                task_id="T5",
                processors=4,
                work_amount_segment=70.21432268421937,
            ),
            Segment(
                segment_id="s16",
                task_id="T5",
                processors=5,
                work_amount_segment=129.78567731578065,
            ),
        ],
    }
    segment_order = [
        "s11",
        "s16",
        "s12",
        "s8",
        "s15",
        "s7",
        "s6",
        "s4",
        "s17",
        "s18",
        "s14",
    ]
    # segments_map, segment_order, _ = simulated_annealing(
    #     instance_data or {},
    #     initial_temperature=50,
    #     final_temperature=5,
    #     cooling_rate=0.98,
    #     iterations_per_temperature=50,
    #     initial_segments_per_task=2,
    # )

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
