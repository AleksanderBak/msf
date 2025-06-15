import random

from msf.algorithms.h1 import Task, h1_algorithm
from msf.algorithms.simulated_annealing import (
    SegmentScheduler,
    load_instance_data,
    simulated_annealing,
)
from msf.algorithms.upperbound import upperbound
from msf.utils.helpers import (
    FinalSchedule,
    group_intervals,
    pretty_print_schedule,
    read_instance_data,
    read_schdeule_from_file,
    save_schedule,
)
from msf.utils.validator import validate_schedule
from msf.visualizer import Visualizer

if __name__ == "__main__":
    random.seed(42)

    data = read_instance_data()
    n_tasks, m_processors, task_info_map = load_instance_data(data)

    final_schedule = simulated_annealing(
        instance_data=data,
        initial_temperature=50,
        final_temperature=40,
        cooling_rate=0.98,
        iterations_per_temperature=1,
        initial_segments_per_task=2,
    )

    res = validate_schedule(final_schedule, data or {})

    if res:
        visualizer = Visualizer()
        visualizer.visualize_schedule_by_processor(
            final_schedule,
            save_to_file=True,
            filename="test_pray_to_god.png",
        )
