import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from src.settings import settings

PLOT_DIR = settings.visualization_dir


class Visualizer:
    def __init__(self):
        pass

    def visualize_schedule_by_processor(
        self,
        schedule_log: list[dict],
        m: int,
        makespan: float,
        filename: str = "gantt_chart_by_processor.png",
        show: bool = False,
    ) -> None:
        """
        Generates a Gantt chart where the Y-axis represents individual processors.

        Args:
            schedule_log (list): List of schedule interval dictionaries.
                                Each dict: {'task_id', 'start_time', 'end_time', 'num_processors'}
            m (int): The total number of processors.
            makespan (float): The total makespan of the schedule.
            filename (str): Name of the file to save the plot.
        """
        if not schedule_log:
            logger.warning("Schedule log is empty, cannot generate visualization.")
            return

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
        tasks = sorted(list(set(item["task_id"] for item in schedule_log)))
        colors = list(mcolors.TABLEAU_COLORS.values())
        task_colors = {
            task_id: colors[i % len(colors)] for i, task_id in enumerate(tasks)
        }
        task_colors["idle"] = "lightgrey"

        for i in range(len(unique_times) - 1):
            t_start = unique_times[i]
            t_end = unique_times[i + 1]
            duration = t_end - t_start

            if duration <= 1e-9:
                continue

            active_tasks_in_interval = []
            for entry in schedule_log:
                if entry["start_time"] < t_end and entry["end_time"] > t_start:
                    active_tasks_in_interval.append(
                        {
                            "task_id": entry["task_id"],
                            "num_processors": entry["num_processors"],
                        }
                    )

            current_processor_id = 0
            processors_used_in_interval = set()

            active_tasks_in_interval.sort(key=lambda x: x["task_id"])

            for task_info in active_tasks_in_interval:
                task_id = task_info["task_id"]
                num_procs = task_info["num_processors"]

                for p_offset in range(num_procs):
                    processor_to_assign = current_processor_id + p_offset
                    plot_data.append(
                        {
                            "processor_id": processor_to_assign,
                            "start_time": t_start,
                            "end_time": t_end,
                            "task_id": task_id,
                        }
                    )
                    processors_used_in_interval.add(processor_to_assign)

                current_processor_id += num_procs
                if current_processor_id >= m:
                    break

            for p_id in range(m):
                if p_id not in processors_used_in_interval:
                    plot_data.append(
                        {
                            "processor_id": p_id,
                            "start_time": t_start,
                            "end_time": t_end,
                            "task_id": "idle",
                        }
                    )

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

        if show:
            plt.show()

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
