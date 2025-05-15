import datetime
import json
import logging
import random
from pathlib import Path

from loguru import logger

from src.speed_generator import ProcessingTimesGenerator


class InstanceGenerator:
    def __init__(
        self,
        instance_path: str,
        instance_name_prefix: str = "instance",
    ) -> None:
        self.instance_path = Path(instance_path)
        self.instances_name_prefix = instance_name_prefix
        self.instance_path.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Instance generator initialized. Output path: {self.instance_path.resolve()}"
        )

    def generate_instances(
        self,
        n_start: int,
        m_start: int,
        n_step: int,
        m_step: int,
        number_of_instances: int,
        seed: int | None = None,
    ) -> list[Path]:
        if seed is not None:
            random.seed(seed)
            logger.debug(f"Seed set to {seed}.")

        generated_files = []

        for i in range(1, number_of_instances + 1):
            n = n_start + (i - 1) * n_step
            m = m_start + (i - 1) * m_step

            if m < n:
                logger.warning(
                    f"Number of tasks ({n}) greater than number of processors ({m})."
                )

            tasks = {}

            times_generator = ProcessingTimesGenerator()

            for j in range(1, n + 1):
                tasks[f"task_{j}"] = {
                    "work_amount": round(random.uniform(25.0, 100.0), 4),
                    # "speed_up": ProcessingTimesGenerator.get_linear_speeds(
                    #     m, seed=seed
                    # ),
                    "speed_up_factors": times_generator.get_concave_speeds(
                        n_processors=m
                    ),
                }

            instance = {"n": n, "m": m, "tasks": tasks}

            instance_name = f"{self.instances_name_prefix}_{i}.json"
            filepath = self.instance_path / instance_name

            try:
                with open(filepath, "w") as f:
                    json.dump(instance, f, indent=4)
                logger.info(f"Instance {i} generated and saved to {filepath.resolve()}")
                generated_files.append(filepath)

            except Exception as e:
                logger.error(
                    f"Failed to save instance {i} to {filepath.resolve()}: {e}"
                )

        return generated_files
