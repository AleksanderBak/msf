import random

import matplotlib.pyplot as plt


class ProcessingTimesGenerator:
    def __init__(self):
        pass

    def generate_speedup_factors(
        self,
        n: int,
        reach_plateau: bool = False,
        initial_plateau_threshold: float = 1,
        final_plateau_threshold: float = 0.97,
    ) -> list[float]:
        speedup_factors = [1.0]
        is_plateaued = False
        plateau_value = 0.0

        for i in range(1, n):
            prev_factor = speedup_factors[-1]

            if reach_plateau and not is_plateaued:
                progress = (i - 1) / max(1, n - 2)
                current_threshold = (
                    initial_plateau_threshold
                    + (final_plateau_threshold - initial_plateau_threshold) * progress
                )
                if random.random() > current_threshold:
                    is_plateaued = True
                    plateau_value = prev_factor

            if is_plateaued:
                next_factor = plateau_value
            else:
                potential_upper_bound = prev_factor + 1
                next_factor = random.uniform(prev_factor, potential_upper_bound)

            speedup_factors.append(next_factor)

        return speedup_factors


if __name__ == "__main__":
    n = 400
    generator = ProcessingTimesGenerator()
    speedup_factors = generator.generate_speedup_factors(
        n=n,
        reach_plateau=True,
    )
    speedup_factors = [0.0] + speedup_factors

    linear_speedup_factors = [i for i in range(n + 1)]

    print(speedup_factors)
    plt.plot(speedup_factors)
    plt.plot(linear_speedup_factors)
    plt.show()
