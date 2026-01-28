import re
import math
import numpy as np
from verl.reward_func_new.executor import PythonExecutor
from verl.reward_func_new.content_utils import extract_code_block, extract_efficiency_metrics


# def check_format(processed_str):
#     has_strategy = ("</strategy>" in processed_str)
#     has_modeling = ("<modeling>" in processed_str) and ("</modeling>" in processed_str)
#     has_check = ("<check>" in processed_str) and ("</check>" in processed_str)
#     if not has_strategy and not has_modeling and not has_check:
#         return False
    
#     final_python_match = re.search(r"</think>.*?```python\s*(.*?)```", processed_str, re.DOTALL)
#     if not final_python_match:
#         return False
    
#     return True


def outcome_reward(obj_val, report, ground_truth, has_code):
    if not has_code:
        return 0.0
    if report != "Done":
        return 0.0
    abs_err = np.abs(obj_val - ground_truth) / (np.abs(ground_truth) + 1.0)
    if abs_err <= 1e-2:
        return 1.0
    else:
        return 0.0


# def add_forced_prefix(solution_str):
#     forced_prefix = "<think>\n<strategy>"
#     if solution_str and solution_str.strip().startswith("<think>"):
#         return solution_str
#     else:
#         return forced_prefix + (solution_str or "")


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, executor_timeout=10):
    batch_size = len(solution_strs)
    executor = PythonExecutor(timeout_length=executor_timeout)
    
    outcome_codes = []
    # full_solutions = []
    
    for i in range(batch_size):
        sol = solution_strs[i]
        # full_sol = add_forced_prefix(sol)
        # full_solutions.append(full_sol)
        
        code = extract_code_block(sol)
        outcome_codes.append(code)

    clean_outcome_codes = [c if c else "" for c in outcome_codes]
    out_obj_results, _, out_code_reports, out_logs = executor.batch_apply(clean_outcome_codes)

    rewards = []
    for i in range(batch_size):
        ground_truth = ground_truths[i] if i < len(ground_truths) else None
        # full_solution = full_solutions[i]

        # if not check_format(full_solution):
        #     rewards.append(0.0)
        #     continue

        outcome_score = outcome_reward(
            obj_val=out_obj_results[i],
            report=out_code_reports[i],
            ground_truth=ground_truth,
            has_code=(outcome_codes[i] is not None)
        )

        rewards.append(outcome_score)
    
    return rewards
