import re
import gurobipy as gp
from gurobipy import GRB


def insert_print(code: str) -> str:
    if "gp.setParam('OutputFlag'" not in code and "gp.setParam(\"OutputFlag\"" not in code:
        code = "gp.setParam('OutputFlag', 1)\n" + code
    
    model_pattern = r'^(\s*)(\w+)\.(optimize)\(\)'
    match = re.search(model_pattern, code, re.MULTILINE)
    if not match:
        return code
    indent = match.group(1)
    model_name = match.group(2)
    pattern = rf'^(\s*)({model_name}\.optimize\(\))'
    replacement = (
        f"{indent}{model_name}.optimize()\n"
        f"{indent}if {model_name}.status == GRB.OPTIMAL:\n"
        f"{indent}    print(f'Just print the best obj: {{{model_name}.ObjVal}}')\n"
        f"{indent}    print('Just print the best sol:[', end = '')\n"
        f"{indent}    for var in {model_name}.getVars():\n"
        f"{indent}        print(f'{{var.X}}', end = ',')\n"
        f"{indent}    print(']')\n"
        f"{indent}else:\n"
        f"{indent}    print('No optimal solution found, status:', {model_name}.status)"
    )
    return re.sub(pattern, replacement, code, flags=re.MULTILINE)


def extract_code_block(response_text: str) -> str:
    if not response_text:
        return None
    after_think = response_text.split('</think>', 1)[-1]
    match = re.search(r"```python\s*(.*?)```", after_think or "", re.DOTALL)
    if match:
        code = match.group(1).strip()
        return insert_print(code)
    else:
        return None


def extract_obj(str_log):
    if 'Just print the best obj:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best obj:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None


def extract_sol(str_log):
    if 'Just print the best sol:' in str_log:
        sol_match = re.search(r'Just print the best sol:\s*\[([-\d.,\s]*)\]', str_log)
        best_sol = [float(x) for x in sol_match.group(1).split(',') if x.strip()] if sol_match else None
        if best_sol:
            best_sol.sort()
            return best_sol
        else:
            return [None]
    return [None]


def extract_efficiency_metrics(log_text, is_milp_detected=False, computed_gap=None):
    # 解析gap
    gap = computed_gap
    if gap is None:
        gap_match = re.search(r"LPGap\s*=\s*([\d\.]+)", log_text)
        if gap_match:
            try:
                gap = float(gap_match.group(1))
            except (ValueError, AttributeError):
                pass
    if gap is None:
        gap_match = re.search(r"Gap\s*=\s*([\d\.]+)%?", log_text)
        if gap_match:
            try:
                val = float(gap_match.group(1))
                gap = val / 100 if val > 1 else val
            except (ValueError, AttributeError):
                pass
    
    # 解析nodes
    nodes = None
    nodes_match = re.search(r"Nodes\s*=\s*(\d+)", log_text)
    if nodes_match:
        try:
            nodes = int(nodes_match.group(1))
        except (ValueError, AttributeError):
            pass
    if nodes is None:
        nodes_match = re.search(r"Explored\s+(\d+)\s+nodes", log_text)
        if nodes_match:
            try:
                nodes = int(nodes_match.group(1))
            except (ValueError, AttributeError):
                pass
    
    # 解析iter_count
    iter_count = None
    iter_match = re.search(r"Solved\s+in\s+(\d+)\s+iterations", log_text, re.IGNORECASE)
    if iter_match:
        try:
            iter_count = int(iter_match.group(1))
        except (ValueError, AttributeError):
            pass
    if iter_count is None:
        iter_match = re.search(r"Iterations\s*:\s*(\d+)", log_text, re.IGNORECASE)
        if iter_match:
            try:
                iter_count = int(iter_match.group(1))
            except (ValueError, AttributeError):
                pass
    if iter_count is None:
        iter_match = re.search(r"Barrier\s+iterations\s*:\s*(\d+)", log_text, re.IGNORECASE)
        if iter_match:
            try:
                iter_count = int(iter_match.group(1))
            except (ValueError, AttributeError):
                pass
    
    # 判断是否为MILP
    is_milp = is_milp_detected or (nodes is not None) or (gap is not None)
    
    return {
        'gap': gap,
        'nodes': nodes,
        'iter_count': iter_count,
        'is_milp': is_milp
    }
