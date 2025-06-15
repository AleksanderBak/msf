import random

from msf.algorithms.h1 import h1_algorithm
from msf.algorithms.simulated_annealing import (
    load_instance_data,
)
from msf.utils.helpers import (
    read_instance_data,
)
from msf.utils.validator import validate_schedule
from msf.visualizer import Visualizer

if __name__ == "__main__":
    random.seed(42)

    data = read_instance_data("instance_1.json")
    n_tasks, m_processors, task_info_map = load_instance_data(data)

    final_schedule = h1_algorithm(
        m_processors,
        n_tasks,
        data["tasks"],
    )
    # print(final_schedule.schedule)
    # simulated_annealing(
    #     instance_data=data,
    #     initial_temperature=50,
    #     final_temperature=40,
    #     cooling_rate=0.98,
    #     iterations_per_temperature=1,
    #     initial_segments_per_task=2,
    # )

    # res = validate_schedule(final_schedule, data or {})

    # if res:
    visualizer = Visualizer()
    visualizer.visualize_schedule_by_processor(
        final_schedule,
        save_to_file=True,
        filename="test_h1.png",
    )
