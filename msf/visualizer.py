import collections
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from msf.config.settings import settings
from msf.utils.models import FinalSchedule

PLOT_DIR = settings.visualization_dir


class Visualizer:
    def __init__(self):
        pass

    def _create_consistent_interval(
        self,
        prev_interval_data,
        current_task_requirements,
        t_start,
        t_end,
    ):
        if not prev_interval_data or not prev_interval_data[0]:
            new_schedule = []
            next_proc_id = 0
            for task_id, count in sorted(current_task_requirements.items()):
                for _ in range(count):
                    new_schedule.append(
                        {
                            "processor_id": next_proc_id,
                            "start_time": t_start,
                            "end_time": t_end,
                            "task_id": task_id,
                        }
                    )
                    next_proc_id += 1
            return [new_schedule]

        last_schedule = prev_interval_data[0]
        previous_assignments = collections.defaultdict(list)
        all_previous_procs = set()
        for assignment in last_schedule:
            previous_assignments[assignment["task_id"]].append(
                assignment["processor_id"]
            )
            all_previous_procs.add(assignment["processor_id"])

        for task_id in previous_assignments:
            previous_assignments[task_id].sort()

        new_assignments = {}
        available_processors = []
        processors_to_keep = set()

        for task_id, required_count in current_task_requirements.items():
            if task_id in previous_assignments:
                old_procs = previous_assignments[task_id]
                procs_to_use = old_procs[:required_count]
                procs_to_free = old_procs[required_count:]
                new_assignments[task_id] = procs_to_use
                processors_to_keep.update(procs_to_use)
                available_processors.extend(procs_to_free)

        freed_by_task_disappearance = all_previous_procs - processors_to_keep
        available_processors.extend(list(freed_by_task_disappearance))
        available_processors.sort()

        for task_id, required_count in current_task_requirements.items():
            needed_more = required_count - len(new_assignments.get(task_id, []))
            if needed_more > 0:
                procs_to_add = available_processors[:needed_more]
                available_processors = available_processors[needed_more:]
                new_assignments.setdefault(task_id, []).extend(procs_to_add)

        next_interval_schedule = []
        for task_id, proc_list in new_assignments.items():
            for proc_id in proc_list:
                next_interval_schedule.append(
                    {
                        "processor_id": proc_id,
                        "start_time": t_start,
                        "end_time": t_end,
                        "task_id": task_id,
                    }
                )
        next_interval_schedule.sort(key=lambda x: x["processor_id"])
        return [next_interval_schedule]

    def visualize_schedule_by_processor(
        self,
        schedule: FinalSchedule,
        save_to_file: bool = True,
        filename: str = "gantt_chart_by_processor.png",
    ) -> None:
        schedule_log = schedule.schedule
        makespan = schedule.makespan
        m = schedule.num_processors

        # --- 1. Identify Time Intervals ---
        time_points = set([0.0, makespan])
        for entry in schedule_log:
            if 0.0 <= entry["start_time"] <= makespan:
                time_points.add(entry["start_time"])
            if 0.0 <= entry["end_time"] <= makespan:
                time_points.add(entry["end_time"])

        sorted_times = sorted(list(time_points))

        # Filter out times very close together to avoid tiny intervals
        unique_times = []
        if sorted_times:
            unique_times.append(sorted_times[0])
            for i in range(1, len(sorted_times)):
                if sorted_times[i] - sorted_times[i - 1] > 1e-9:
                    unique_times.append(sorted_times[i])

        if not unique_times or (len(unique_times) == 1 and unique_times[0] == 0.0):
            logger.warning(
                "No valid time intervals found in the schedule log. "
                "Ensure that the log contains valid start and end times."
            )
            return

        plot_data = []
        unique_tasks = sorted({item["task_id"] for item in schedule_log})
        colors = list(mcolors.TABLEAU_COLORS.values())
        task_colors = dict(zip(unique_tasks, colors[: len(unique_tasks)]))
        task_colors["idle"] = "lightgrey"

        prev_interval_data = []
        intervals = []

        for i in range(len(unique_times) - 1):
            t_start, t_end = unique_times[i], unique_times[i + 1]

            if t_end - t_start <= 1e-9:
                continue

            current_task_requirements = collections.defaultdict(int)
            for entry in schedule_log:
                if entry["start_time"] <= t_start and entry["end_time"] >= t_end:
                    current_task_requirements[entry["task_id"]] += entry[
                        "num_processors"
                    ]

            if not current_task_requirements:
                prev_interval_data = []
                continue

            next_interval = self._create_consistent_interval(
                prev_interval_data,
                current_task_requirements,
                t_start,
                t_end,
            )
            prev_interval_data = next_interval

            used_processors = set()
            for assignment in next_interval[0]:
                used_processors.add(assignment["processor_id"])

            plot_interval = []
            for processor in range(m):
                if processor not in used_processors:
                    plot_interval.append(
                        {
                            "processor_id": processor,
                            "start_time": t_start,
                            "end_time": t_end,
                            "task_id": "idle",
                        }
                    )
            plot_interval.extend(next_interval[0])

            plot_data.extend(plot_interval)
            # print("Next interval:")
            # for interval in next_interval:
            #     print(interval)
            # print("-" * 100)

        # --- 4. Plot ---
        fig, ax = plt.subplots(
            figsize=(12, max(4, m * 0.3))
        )  # Adjust figure size based on 'm'

        sns.set_theme(style="whitegrid", palette="deep")

        for item in plot_data:
            processor = item["processor_id"]
            start = item["start_time"]
            end = item["end_time"]
            task = item["task_id"]
            duration = end - start
            color = task_colors.get(
                task, "black"
            )  # Default to black if task color not found

            # Draw the rectangle for this processor/time slot
            ax.barh(
                y=processor,
                width=duration,
                left=start,
                height=1.0,  # Height 1 fills the row
                color=color,
                edgecolor="black",
                alpha=0.9,
                linewidth=0.5,
            )

            # Optional: Add task label inside the rectangle if it's large enough
            if (
                duration > makespan * 0.03 and processor % max(1, m // 5) == 0
            ):  # Heuristics to reduce clutter
                # Center text - adjustments might be needed for aesthetics
                text_x = start + duration / 2
                text_y = processor
                # Don't label idle time unless necessary
                if task != "idle":
                    ax.text(
                        text_x,
                        text_y,
                        task,
                        ha="center",
                        va="center",
                        color="white"
                        if sum(mcolors.to_rgb(color)) < 1.5
                        else "black",  # Contrast
                        fontweight="bold",
                        fontsize=7,
                    )

        # Configure the plot
        ax.set_yticks(np.arange(m))
        ax.set_yticklabels([str(i) for i in range(m)])
        ax.set_ylim(-0.5, m - 0.5)  # Set Y limits tightly around processors

        ax.set_xlabel("Time")
        ax.set_ylabel("Processor ID")
        ax.set_title("Processor Allocation Schedule")

        # Set x-axis limits
        ax.set_xlim(0, makespan * 1.01)  # Add a small buffer

        # Add grid lines for readability
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        # Y-axis: Processor 0 at the bottom (default behavior, no inversion needed)

        plt.tight_layout()  # Adjust layout

        try:
            plt.savefig(os.path.join(PLOT_DIR, filename))
        except Exception as e:
            print(f"Error saving processor-centric schedule chart: {e}")

    def visualize_schedule(
        self,
        schedule_log,
        makespan,
        filename="gantt_chart.png",
        show: bool = False,
    ):
        """
        Generates a Gantt chart from the schedule log using Matplotlib.

        Args:
            schedule_log (list): List of schedule interval dictionaries.
            makespan (float): The total makespan of the schedule.
            filename (str): Name of the file to save the plot.
        """
        if not schedule_log:
            print("Schedule log is empty, cannot generate visualization.")
            return

        tasks = sorted(list(set(item["task_id"] for item in schedule_log)))
        num_tasks = len(tasks)

        fig, ax = plt.subplots(
            figsize=(12, max(6, num_tasks * 0.5))
        )  # Adjust figure size

        # Assign a unique y-coordinate (row) to each task
        task_y_map = {task_id: i for i, task_id in enumerate(tasks)}

        # Generate distinct colors for tasks
        # Using a predefined list and cycling through it
        colors = list(mcolors.TABLEAU_COLORS.values())  # Get tableau colors
        task_colors = {
            task_id: colors[i % len(colors)] for i, task_id in enumerate(tasks)
        }

        # Plot each interval as a horizontal bar
        for entry in schedule_log:
            task_id = entry["task_id"]
            start = entry["start_time"]
            end = entry["end_time"]
            num_procs = entry["num_processors"]
            duration = end - start

            # Skip plotting if duration is zero or negligible
            if duration < 1e-6:
                continue

            y_pos = task_y_map[task_id]
            color = task_colors[task_id]

            # Draw the bar
            ax.barh(
                y=y_pos,
                width=duration,
                left=start,
                height=0.6,
                color=color,
                edgecolor="black",
                alpha=0.8,
            )

            # Add text inside the bar (number of processors)
            text_x = start + duration / 2
            text_y = y_pos
            # Only add text if the bar is wide enough
            if duration > makespan * 0.02:  # Heuristic threshold
                ax.text(
                    text_x,
                    text_y,
                    f"P={num_procs}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=8,
                )

        # Configure the plot
        ax.set_yticks(range(num_tasks))
        ax.set_yticklabels(tasks)
        ax.set_xlabel("Time")
        ax.set_ylabel("Task ID")
        ax.set_title("Malleable Task Schedule Gantt Chart")

        # Set x-axis limits
        ax.set_xlim(0, makespan * 1.05)  # Add a little buffer

        # Add grid lines for readability
        ax.grid(True, axis="x", linestyle=":", alpha=0.7)

        # Invert y-axis so T1 (or first task alphabetically) is at the top
        ax.invert_yaxis()

        plt.tight_layout()  # Adjust layout to prevent labels overlapping

        try:
            plt.savefig(os.path.join(PLOT_DIR, filename))
        except Exception as e:
            print(f"Error saving Gantt chart: {e}")

        if show:
            plt.show()
