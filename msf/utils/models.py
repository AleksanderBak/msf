import json
import os

from loguru import logger

from msf.config.settings import settings


class FinalSchedule:
    def __init__(
        self, makespan: float, schedule: list, num_processors: int, num_tasks: int
    ):
        self.makespan = makespan
        self.schedule = schedule
        self.num_processors = num_processors
        self.num_tasks = num_tasks

    def save_to_file(self, filename: str):
        directory = settings.schedules_dir
        filename = os.path.join(directory, filename)

        schedule = {
            "num_processors": self.num_processors,
            "num_tasks": self.num_tasks,
            "makespan": self.makespan,
            "schedule": self.schedule,
        }
        try:
            with open(filename, "w") as f:
                json.dump(schedule, f, indent=2)
                logger.info(f"Schedule saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save schedule to {filename}: {e}")
