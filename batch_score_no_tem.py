import re
import math
import numpy as np
from verl.reward_func_new.executor import PythonExecutor
from verl.reward_func_new.content_utils import extract_code_block, extract_efficiency_metrics


def format_reward(processed_str):
    think_match = re.search(r"<think>(.*?)</think>", processed_str, re.DOTALL)
    if not think_match:
        return 0.0
    think_content = think_match.group(1).strip()
    if not think_content:
        return 0.0
    code_match = re.search(r"</think>.*?```python\s*(.*?)```", processed_str, re.DOTALL)
    if not code_match:
        return 0.0
    return 1.0


def outcome_reward(obj_val, report, ground_truth, has_code):
    if not has_code:
        return 0.0
    if report != "Done":
        return 0.0
    if obj_val is None:
        return 0.2
    abs_err = np.abs(obj_val - ground_truth) / (np.abs(ground_truth) + 1.0)
    if abs_err <= 1e-2:
        return 1.0
    else:
        return 0.4


def efficiency_reward(metrics):
    gap = metrics.get('gap')
    nodes = metrics.get('nodes')
    iter_count = metrics.get('iter_count')
    is_milp = metrics.get('is_milp', False)
    
    if is_milp:
        # MILP: 使用gap和nodes
        r_gap, r_node = 0.0, 0.0
        if gap is not None and gap >= 0.0:
            gap = max(gap, 0.0)
            r_gap = 1.0 - math.tanh(3.0 * gap)
        if nodes is not None and nodes >= 0:
            nodes = max(nodes, 0)
            r_node = 1.0 - math.tanh(nodes / 5.0)
        if gap is None and nodes is None:
            return 0.0
        r_bonus = r_node if gap is None else (r_gap if nodes is None else 0.5 * r_gap + 0.5 * r_node)
    else:
        # LP: 使用迭代次数
        if iter_count is None or iter_count < 0:
            return 0.0
        iter_count = max(iter_count, 0)
        r_bonus = 1.0 - math.tanh(iter_count / 15.0)
    
    return max(0.0, min(1.0, r_bonus))


def add_forced_prefix(solution_str):
    forced_prefix = "<think>\n<strategy>"
    if solution_str and solution_str.strip().startswith("<think>"):
        return solution_str
    else:
        return forced_prefix + (solution_str or "")


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, executor_timeout=10):
    batch_size = len(solution_strs)
    executor = PythonExecutor(timeout_length=executor_timeout)
    
    outcome_codes = []
    full_solutions = []
    
    for i in range(batch_size):
        sol = solution_strs[i]
        full_sol = add_forced_prefix(sol)
        full_solutions.append(full_sol)
        
        code = extract_code_block(full_sol)
        outcome_codes.append(code)

    clean_outcome_codes = [c if c else "" for c in outcome_codes]
    out_obj_results, _, out_code_reports, out_logs = executor.batch_apply(clean_outcome_codes)

    rewards = []
    for i in range(batch_size):
        ground_truth = ground_truths[i] if i < len(ground_truths) else None
        full_solution = full_solutions[i]

        format_score = format_reward(full_solution)
        if format_score <= 0.0:
            rewards.append(0.0)
            continue

        outcome_score = outcome_reward(
            obj_val=out_obj_results[i],
            report=out_code_reports[i],
            ground_truth=ground_truth,
            has_code=(outcome_codes[i] is not None)
        )
        
        # 计算效率奖励（仅在答案正确时）
        efficiency_score = 0.0
        if outcome_score == 1.0:
            log_text = out_logs[i] if i < len(out_logs) else ""
            metrics = extract_efficiency_metrics(log_text)
            efficiency_score = efficiency_reward(metrics)

        rewards.append(0.2 * format_score + outcome_score + 0.5 * efficiency_score)
    
    return rewards
