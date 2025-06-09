import random

import matplotlib.pyplot as plt


class ProcessingTimesGenerator:
    def __init__(self):
        pass

    def generate_speedup_factors(
        self,
        n: int,
        reach_plateau: bool = False,
        initial_plateau_threshold: float = 0.7,
        final_plateau_threshold: float = 0.9,
        growth_aggressiveness: float = 0.1,
    ) -> list[float]:
        speedup_factors = [1.0]
        is_plateaued = False
        plateau_value = 0.0

        for i in range(1, n):
            prev_factor = speedup_factors[-1]

            if reach_plateau and not is_plateaued:
                progress = (i - 1) / max(1, n - 2)
                print(f"Progress: {progress}")
                current_threshold = (
                    initial_plateau_threshold
                    + (final_plateau_threshold - initial_plateau_threshold) * progress
                )

                print(f"Current threshold: {current_threshold}")
                if random.random() > current_threshold:
                    is_plateaued = True
                    plateau_value = prev_factor

            if is_plateaued:
                next_factor = plateau_value
            else:
                potential_upper_bound = i + 1
                upper_bound = min(
                    prev_factor
                    + (potential_upper_bound - prev_factor) * growth_aggressiveness,
                    potential_upper_bound,
                )

                next_factor = random.uniform(prev_factor, upper_bound)

            speedup_factors.append(next_factor)

        return speedup_factors


if __name__ == "__main__":
    n = 20
    generator = ProcessingTimesGenerator()
    speedup_factors = generator.generate_speedup_factors(
        n=n,
        reach_plateau=True,
        initial_plateau_threshold=0.95,
        final_plateau_threshold=0.99,
        growth_aggressiveness=1,
    )
    speedup_factors = [0.0] + speedup_factors

    linear_speedup_factors = [i for i in range(n + 1)]

    plt.plot(speedup_factors)
    plt.plot(linear_speedup_factors)
    plt.show()
