from msf.algorithms.simulated_annealing import Segment

segments_map = {
    "T1": [
        Segment(
            segment_id="s2069",
            task_id="T1",
            processors=2,
            work_amount_segment=99.99999999999997,
        )
    ],
    "T2": [
        Segment(
            segment_id="s2039", task_id="T2", processors=2, work_amount_segment=80.0
        )
    ],
    "T3": [
        Segment(
            segment_id="s2074",
            task_id="T3",
            processors=4,
            work_amount_segment=120.0,
        )
    ],
    "T4": [
        Segment(
            segment_id="s2027", task_id="T4", processors=3, work_amount_segment=50.0
        )
    ],
    "T5": [
        Segment(
            segment_id="s2043",
            task_id="T5",
            processors=5,
            work_amount_segment=4.88965119840482,
        ),
        Segment(
            segment_id="s2076",
            task_id="T5",
            processors=1,
            work_amount_segment=15.59342660696153,
        ),
        Segment(
            segment_id="s2080",
            task_id="T5",
            processors=1,
            work_amount_segment=16.675169334339135,
        ),
        Segment(
            segment_id="s2082",
            task_id="T5",
            processors=2,
            work_amount_segment=1.1400544832596542,
        ),
        Segment(
            segment_id="s2084",
            task_id="T5",
            processors=3,
            work_amount_segment=15.32479524940029,
        ),
        Segment(
            segment_id="s2085",
            task_id="T5",
            processors=1,
            work_amount_segment=1.1687161883894757,
        ),
        Segment(
            segment_id="s2089",
            task_id="T5",
            processors=1,
            work_amount_segment=12.7473800566725,
        ),
        Segment(
            segment_id="s2093",
            task_id="T5",
            processors=2,
            work_amount_segment=37.86728882631493,
        ),
        Segment(
            segment_id="s2094",
            task_id="T5",
            processors=6,
            work_amount_segment=22.09775038719311,
        ),
        Segment(
            segment_id="s2098",
            task_id="T5",
            processors=6,
            work_amount_segment=15.988348761186263,
        ),
        Segment(
            segment_id="s2100",
            task_id="T5",
            processors=3,
            work_amount_segment=23.166713719539487,
        ),
        Segment(
            segment_id="s2101",
            task_id="T5",
            processors=2,
            work_amount_segment=16.954585219710154,
        ),
        Segment(
            segment_id="s2102",
            task_id="T5",
            processors=4,
            work_amount_segment=16.386119968628705,
        ),
    ],
}

segment_order = [
    "s2093",
    "s2094",
    "s2080",
    "s2039",
    "s2085",
    "s2074",
    "s2089",
    "s2082",
    "s2100",
    "s2101",
    "s2102",
    "s2043",
    "s2098",
    "s2084",
    "s2027",
    "s2076",
    "s2069",
]


# def decode_solution_original(
#     solution_task_segments_map: dict[str, list[Segment]],
#     global_segment_order_ids: list[str],
#     m_total_processors: int,
#     task_info_map: dict[str, Any],
# ) -> tuple[list[ScheduledSegmentOutput], float] | tuple[None, None]:
#     """
#     Constructs a schedule from the solution representation.

#     Args:
#         solution_task_segments_map: dict[str, list[Segment]] - Solution task segments map.
#         global_segment_order_ids: list[str] - Global segment order IDs.
#         m_total_processors: int - Total number of processors.
#         task_info_map: dict[str, Any] - Task information map.

#     Returns:
#         tuple[list[ScheduledSegmentOutput], float] - Scheduled outputs and makespan.
#     """
#     scheduled_outputs = []
#     makespan = 0.0

#     # processor_finish_times[i] = time when physical system processor 'i' becomes free
#     processor_finish_times = [0.0] * m_total_processors

#     segments_to_schedule = [
#         segment
#         for _, segment_list in solution_task_segments_map.items()
#         for segment in segment_list
#     ]

#     for segment_to_place in segments_to_schedule:
#         segment_to_place.duration = get_segment_duration(
#             segment_to_place, task_info_map
#         )

#         if (
#             segment_to_place.duration == float("inf")
#             or segment_to_place.processors <= 0
#             or segment_to_place.processors > m_total_processors
#         ):
#             logger.error(
#                 f"Invalid segment: {segment_to_place}. Returning empty schedule."
#             )
#             return None, None

#         num_procs_needed = segment_to_place.processors

#         earliest_start_time = 0.0
#         assigned_proc_indices = []

#         # Find the earliest time this segment can start (Greedy Approach)
#         # Iterate through all unique processor finish times as potential start points
#         # This ensures we check event points where processor availability might change.
#         distinct_finish_times = sorted(list(set(processor_finish_times)))
#         candidate_start_times = [0.0] + [
#             t for t in distinct_finish_times if t > 1e-9
#         ]  # Add 0 and positive finish times

#         best_t_start_for_segment = float("inf")

#         for t_candidate in candidate_start_times:
#             available_procs_at_t_candidate = []
#             for proc_idx in range(m_total_processors):
#                 if (
#                     processor_finish_times[proc_idx] <= t_candidate + 1e-9
#                 ):  # Check if proc is free by t_candidate
#                     available_procs_at_t_candidate.append(proc_idx)

#             if len(available_procs_at_t_candidate) >= num_procs_needed:
#                 # Found enough processors. This t_candidate is a valid start time.
#                 best_t_start_for_segment = t_candidate
#                 assigned_proc_indices = available_procs_at_t_candidate[
#                     :num_procs_needed
#                 ]
#                 break  # Found the earliest possible, due to sorted candidate_start_times

#         if not assigned_proc_indices:  # Should only happen if num_procs_needed > m_total_processors (checked) or logic error
#             # print(f"Decoder Error: Could not find slot for segment {segment_to_place}")
#             return [], float("inf")

#         earliest_start_time = best_t_start_for_segment
#         segment_end_time = earliest_start_time + segment_to_place.duration

#         # Update finish times for the assigned physical processors
#         for proc_idx in assigned_proc_indices:
#             processor_finish_times[proc_idx] = segment_end_time

#         scheduled_outputs.append(
#             ScheduledSegmentOutput(
#                 task_id=segment_to_place.task_id,
#                 start_time=earliest_start_time,
#                 end_time=segment_end_time,
#                 num_processors=num_procs_needed,
#             )
#         )
#         makespan = max(makespan, segment_end_time)

#     return scheduled_outputs, makespan
