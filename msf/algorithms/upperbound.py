from msf.utils.helpers import FinalSchedule


def upperbound(instance):
    """
    Simple upperbound scheduler that assigns all processors to each task
    and executes tasks sequentially.
    """
    m = instance["m"]  # number of processors
    tasks = instance["tasks"]

    schedule = []
    current_time = 0.0

    # Process each task sequentially using all processors
    for task_id, task_data in tasks.items():
        work_amount = task_data["work_amount"]
        speed_up_factor = task_data["speed_up_factors"][m - 1]  # Use all m processors

        # Calculate execution time: work_amount / speed_up_factor
        execution_time = work_amount / speed_up_factor

        schedule.append(
            {
                "task_id": task_id,
                "start_time": current_time,
                "end_time": current_time + execution_time,
                "num_processors": m,
            }
        )

        current_time += execution_time

    makespan = current_time

    result = FinalSchedule(
        makespan=makespan,
        schedule=schedule,
        num_processors=m,
        num_tasks=len(tasks),
    )

    return result
