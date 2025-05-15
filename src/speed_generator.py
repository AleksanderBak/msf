import random


class ProcessingTimesGenerator:
    def check_convexity(self, sequence: list[float]) -> bool:
        """
        Check if the sequence is convex.
        """
        n = len(sequence)
        if n < 3:
            raise ValueError("Sequence must have at least three elements.")

        for i in range(2, n - 1):
            if sequence[i + 1] - 2 * sequence[i] + sequence[i - 1] < 0:
                return False
        return True

    def check_concavity(self, sequence: list[float]) -> bool:
        """
        Check if the sequence is concave.
        """
        n = len(sequence)
        if n < 3:
            raise ValueError("Sequence must have at least three elements.")

        for i in range(1, n - 2):
            if sequence[i + 1] - 2 * sequence[i] + sequence[i - 1] > 0:
                return False
        return True

    def get_linear_speeds(n_processors: int, seed: int) -> list[float]:
        return [float(i) for i in range(1, n_processors + 1)]

    def get_concave_speeds(
        self, n_processors: int, min_time: float = 1.0
    ) -> list[float]:
        max_time = n_processors

        times = [0.0] * n_processors
        alpha = random.uniform(0.1, 0.9)  # Controls curvature
        print("Generated alpha:", alpha)
        times[0] = 1.0

        for k in range(2, n_processors + 1):
            time_k = k**alpha
            print(f"Time for processor {k}: {time_k}")
            times[k - 1] = min(max_time, time_k)

        old_min = min(times)
        old_max = max(times)
        scaled_times = [
            min_time + ((x - old_min) / (old_max - old_min)) * (max_time - min_time)
            for x in times
        ]
        rounded_results = [round(t, 4) for t in scaled_times]

        if self.check_concavity(rounded_results):
            return rounded_results
        else:
            raise ValueError("Generated sequence is not concave.")

    def get_convex_speeds(
        self,
        n_processors: int,
        p1_time: float = 1.0,
        alpha_range: tuple[float, float] = (
            1.1,
            5.0,
        ),  # alpha > 1 ensures convex increasing
        min_increment: float = 0.1,  # Minimum increase between times
        min_time: float = 1.0,  # Minimum possible processing time
    ) -> list[float]:
        max_time = n_processors
        times = [0.0] * n_processors
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        times[0] = p1_time  # Time for k=1

        for k in range(2, n_processors + 1):
            time_k = p1_time * (k**alpha)
            # Ensure strict increase
            time_k = max(time_k, times[k - 2] + min_increment)
            times[k - 1] = time_k

        old_min = min(times)
        old_max = max(times)
        scaled_times = [
            min_time + ((x - old_min) / (old_max - old_min)) * (max_time - min_time)
            for x in times
        ]
        rounded_results = [round(t, 4) for t in scaled_times]

        if self.check_convexity(rounded_results):
            return rounded_results
        else:
            raise ValueError("Generated sequence is not convex.")
