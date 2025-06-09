import random

from msf.algorithms.h1 import Task, h1_algorithm
from msf.algorithms.simulated_annealing import simulated_annealing
from msf.algorithms.upperbound import upperbound
from msf.utils.helpers import (
    FinalSchedule,
    pretty_print_schedule,
    read_instance_data,
    read_schdeule_from_file,
    save_schedule,
)
from msf.visualizer import Visualizer

if __name__ == "__main__":
    random.seed(42)

    # ============= Get Instance Data =============
    data = read_instance_data()
    if data is None:
        raise ValueError("No instance data found")

    upperbound_schedule = upperbound(data)
    upperbound = upperbound_schedule.makespan

    print(f"Upperbound: {upperbound}")

    # ============= Run H1 Algorithm =============
    h1_schedule = h1_algorithm(m=data["m"], n=data["n"], tasks_dict=data["tasks"])

    # ============= Run Simulated Annealing =============
    sa_schedule = simulated_annealing(
        instance_data=data,
        initial_temperature=50,
        final_temperature=5,
        cooling_rate=0.98,
        iterations_per_temperature=50,
        initial_segments_per_task=2,
    )

    # ============= Save Schedule =============
    # save_schedule(h1_schedule, "h1_schedule.json")
    save_schedule(sa_schedule, "sa_schedule.json")

    # ============= Visualize Schedules =============
    visualizer = Visualizer()

    visualizer.visualize_schedule_by_processor(h1_schedule, filename="h1_schedule.png")
    visualizer.visualize_schedule_by_processor(
        sa_schedule, filename="sa_schedule_weird.png"
    )

    visualizer.visualize_schedule_by_processor(
        upperbound_schedule, filename="upperbound_schedule.png"
    )

    # random.seed(42)
    # final_schedule = read_schdeule_from_file("h1_schedule_2.json")

    # pretty_print_schedule("H1", final_schedule)

    # visualizer = Visualizer()

    # visualizer.visualize_schedule_by_processor(
    #     final_schedule, filename="h1_schedule_2_by_processor.png"
    # )

    # simulated_annealing_schedule, simulated_annealing_makespan = simulated_annealing(

    # data = read_instance_data("instance_4.json")

    # m = data["m"]
    # n = data["n"]
    # task_data = data["tasks"]
    # tasks = []

    # for key, val in task_data.items():
    #     speedups = unify_speedup_factors(val["speed_up_factors"], m)
    #     task = Task(key, float(val["work_amount"]), speedups)
    #     tasks.append(task)

    # final_schedule = h1_algorithm(m=m, n=n, tasks=tasks)
    # final_schedule.save_to_file("h1_schedule_2.json")

    # pretty_print_schedule("H1", final_schedule.schedule, final_schedule.makespan)

    # save_schedule(schedule_h1, "h1_schedule.json")

    # visualizer = Visualizer()
    # visualizer.visualize_schedule_by_processor(
    #     schedule_h1,
    #     m,
    #     makespan_h1,
    #     "h1_schedule_by_processor.png",
    #     show=False,
    # )
    # simulated_annealing_schedule, simulated_annealing_makespan = simulated_annealing(
    #     instance_data=data,
    #     initial_temperature=50,
    #     final_temperature=5,
    #     cooling_rate=0.99,
    #     iterations_per_temperature=50,
    # )

    # pretty_print_schedule(
    #     "Simulated Annealing",
    #     simulated_annealing_schedule,
    #     simulated_annealing_makespan,
    # )

    # save_schedule(simulated_annealing_schedule, "simulated_annealing_schedule.json")

    # visualizer.visualize_schedule_by_processor(
    #     simulated_annealing_schedule,
    #     data["m"],
    #     simulated_annealing_makespan,
    #     "simulated_annealing_schedule_by_processor.png",
    #     show=False,
    # )
