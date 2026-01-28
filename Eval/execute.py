import os
import re
import json
import argparse
import subprocess
from collections import defaultdict


ADD_SCRIPT = '''
if model.status == GRB.OPTIMAL:
    print(f"Just print the best solution: {model.objVal}")
else:
    print("No Best Solution")
'''

def check_environment():
    """Check Python version and gurobipy installation"""
    print('=' * 10, "Confirming Python version and gurobipy package installation.", "=" * 10)
    subprocess.run(['python3', '--version'], text=True, check=True)
    subprocess.run(['python3', '-c', 'import gurobipy'], text=True, check=True)
    print('=' * 10, "Check complete. If there are errors, please verify gurobipy installation", "=" * 10)

def extract_code(code_str):
    """Extract Python code from markdown-style string"""
    pattern = r'```python(.*?)```'
    match = re.search(pattern, code_str, re.DOTALL)
    code = None
    if match:
        code = match.group(1)
    else:
        code = code_str
    return (code + ADD_SCRIPT) if code else None

def compile_script(script_content, example_id, timeout=10):
    gurobi_dir = os.path.join(args.output_dir, 'gurobi_code')
    os.makedirs(gurobi_dir, exist_ok=True)
    
    with open(os.path.join(gurobi_dir, f'{example_id}.py'), 'w') as f:
        f.write(script_content)

    try:
        process = subprocess.run(['python3', os.path.join(gurobi_dir, f'{example_id}.py')], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        execution_result = process.stdout
        execution_best_solution = extract_obj(execution_result)
        if execution_best_solution is not None:
            execution_state = "Execution Successful and Best Solution Found"
        elif "No Best Solution" in execution_result:
            execution_best_solution = "No Best Solution"
            execution_state = "Execution Successful but No Best Solution Found"
        else:
            execution_best_solution = None
            execution_state = "Execution Successful but Out of Expectation"
    except subprocess.TimeoutExpired as e:
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        execution_result = str(e)
        execution_best_solution = None
        execution_state = f"Execution Failed: {e.stderr}"

    return {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }

def extract_obj(str_log):
    """Extract objective value from log string"""
    if 'Just print the best solution:' in str_log:
        item = next(i for i in str_log.split('\n') if 'Just print the best solution:' in i)
        result = re.findall(r'-?\d+\.?\d*', item)
        return float(result[0]) if result else None
    return None

def process_example(example, question_field, answer_field, use_percentage_err_tolerance, err_tolerance, index, k):
    example_id = example.get('id', f'{index}')
    difficulty = example.get('difficulty', 'no_difficulty')
    code_field = f"{question_field[:2]}_gurobi_code"
    
    # Assume outputs[key] gives you a list of outputs for the example
    scripts = example.get(code_field, [])
    scripts = [extract_code(i) for i in scripts]

    if not scripts:
        return {
            'id': example_id,
            'difficulty': difficulty,
            f'pass_{k}': 0,
            'results_correct': [False] * k,  # No correct test cases
            'execution_details': [{'state': "Execution Failed: No code", 'best_solution': None}] * k  # Details for each execution
        }

    results = []
    execution_details = []  # Collect details for each execution
    for i, script in enumerate(scripts[:k]):  # Limit to k scripts
        if not script:
            results.append(False)
            execution_details.append({'state': "Execution Failed: No code", 'best_solution': None})
            continue
        
        execution_output = compile_script(script, f"{example_id}_{i}")
        gt_answer = example[answer_field]
        pred_answer = execution_output["execution_best_solution"]
        
        if gt_answer == "No Best Solution":
            result_correct = (pred_answer is not None and pred_answer == gt_answer)
        elif pred_answer is not None and pred_answer != "No Best Solution":
            gt_answer = float(gt_answer) if isinstance(gt_answer, str) else (float(gt_answer[0]) if isinstance(gt_answer, list) else float(gt_answer))
            pred_answer = float(pred_answer)
            if gt_answer == 0:
                result_correct = abs(pred_answer) <= err_tolerance
            else:
                if use_percentage_err_tolerance:
                    result_correct = abs((pred_answer - gt_answer) / gt_answer) <= err_tolerance
                else:
                    result_correct = abs(pred_answer - gt_answer) <= err_tolerance
        else:
            result_correct = False
        
        results.append(result_correct)
        execution_details.append({
            'state': execution_output["execution_state"],
            'best_solution': execution_output["execution_best_solution"]
        })
    
    pass_k = any(results)
    return {
        'id': example_id,
        'difficulty': difficulty,
        f'pass_{k}': pass_k,
        'results_correct': results,
        'execution_details': execution_details
    }

def main(args):
    check_environment()

    with open(args.input_file) as fd:
        data = [json.loads(line) for line in fd]

    k = args.passk
    results = []
    for i, example in enumerate(data):
        try:
            result = process_example(example, args.question_field, args.answer_field, args.use_percentage_err_tolerance, args.err_tolerance, i, k)
            results.append(result)
        except Exception as exc:
            print(f'An error occurred: {exc}')

    # Generate report
    pass_k_count = sum(r[f'pass_{k}'] for r in results)
    total_count = len(results)
    pass_k_accuracy = pass_k_count / total_count if total_count > 0 else 0

    execution_state_counts = defaultdict(int)
    for r in results:
        for d in r['execution_details']:
            execution_state_counts[d['state']] += 1

    easy_correct_count = sum(1 for r in results if r[f'pass_{k}'] and r['difficulty'] in ['Easy', 'easy'])
    easy_total_count = sum(1 for r in results if r['difficulty'] in ['Easy', 'easy'])
    easy_accuracy = easy_correct_count / easy_total_count if easy_total_count > 0 else 0

    medium_correct_count = sum(1 for r in results if r[f'pass_{k}'] and r['difficulty'] in ['Medium', 'medium'])
    medium_total_count = sum(1 for r in results if r['difficulty'] in ['Medium', 'medium'])
    medium_accuracy = medium_correct_count / medium_total_count if medium_total_count > 0 else 0

    hard_correct_count = sum(1 for r in results if r[f'pass_{k}'] and r['difficulty'] in ['Hard', 'hard'])
    hard_total_count = sum(1 for r in results if r['difficulty'] in ['Hard', 'hard'])
    hard_accuracy = hard_correct_count / hard_total_count if hard_total_count > 0 else 0
    
    correct_id = []
    for item in results:
        if item[f'pass_{k}'] == 'true':
            correct_id.append(item['id'])

    report = {
        'accuracy': pass_k_accuracy,
        'correct_count': pass_k_count,
        'total_count': total_count,
        'easy_accuracy': easy_accuracy,
        'easy_correct_count': easy_correct_count,
        'easy_total_count': easy_total_count,
        'medium_accuracy': medium_accuracy,
        'medium_correct_count': medium_correct_count,
        'medium_total_count': medium_total_count,
        'hard_accuracy': hard_accuracy,
        'hard_correct_count': hard_correct_count,
        'hard_total_count': hard_total_count,
        'correct_id':correct_id,
        'execution_state_counts': execution_state_counts,
        'results': results
    }

    # Save report
    with open(os.path.join(args.output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print("accuracy:", pass_k_accuracy)
    print("easy_accuracy:", easy_accuracy)
    print("medium_accuracy:", medium_accuracy)
    print("hard_accuracy:", hard_accuracy)
    print("correct_count:", pass_k_count)
    print("total_count:", total_count)
    print(f"Evaluation report saved to {os.path.join(args.output_dir, 'evaluation_report.json')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for code execution")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker threads")
    parser.add_argument("--question_field", type=str, default="zh_question", help="Field name for questions")
    parser.add_argument("--answer_field", type=str, default="zh_answer", help="Field name for answers")
    parser.add_argument("--passk", type=int, default=1, help="Number of top generations to consider for pass@k")
    parser.add_argument("--use_percentage_err_tolerance", type=bool, default=False, help="Enable percentage error tolerance")
    parser.add_argument("--err_tolerance", type=float, default=0.05, help="Tolerance for error")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
