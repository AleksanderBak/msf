import math
from collections import defaultdict

from loguru import logger

from msf.utils.helpers import FinalSchedule


def validate_schedule(schedule: FinalSchedule, instance_data: dict) -> bool:
    num_processors = schedule.num_processors
    final_makespan = schedule.makespan

    final_work = {task_id: 0 for task_id in instance_data["tasks"]}

    interval_groups = defaultdict(list)
    for task in schedule.schedule:
        interval_key = (task["start_time"], task["end_time"])
        interval_groups[interval_key].append(task)
    sorted_intervals = sorted(interval_groups.items(), key=lambda x: x[0][0])
    makespan = None

    for (start_time, end_time), tasks in sorted_intervals:
        if makespan is None and start_time != 0:
            logger.error(
                f"Start time of the first interval is not equal to 0 ({start_time})"
            )
            return False

        used_processors = 0

        for task in tasks:
            used_processors += task["num_processors"]
            work_amount = (end_time - start_time) * instance_data["tasks"][
                task["task_id"]
            ]["speed_up_factors"][task["num_processors"] - 1]
            final_work[task["task_id"]] += work_amount

        if used_processors > num_processors:
            logger.error(
                f"Used processors in interval {start_time} - {end_time} is {used_processors} > {num_processors}"
            )
            return False

        if makespan is not None and start_time != makespan:
            logger.error(
                f"ERROR: Start time of interval {start_time} is not equal to end time of previous interval {makespan}, there are gaps between intervals"
            )
            return False
        makespan = end_time

    for task_id, work in final_work.items():
        if not math.isclose(work, instance_data["tasks"][task_id]["work_amount"]):
            logger.error(
                f"ERROR: Work amount for task {task_id} is {work} != {instance_data['tasks'][task_id]['work_amount']}"
            )
            return False

    if makespan is None:
        logger.error("Final makespan is None")
        return False

    if not math.isclose(final_makespan, makespan):
        logger.error(
            f"ERROR: Final makespan is {final_makespan} != {makespan}, there are gaps between intervals"
        )
        return False

    logger.success("Schedule is valid")
    return True
