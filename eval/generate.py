import os
import re
import json
import argparse
from typing import Dict, List

from tqdm import tqdm
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


TEMPLATES = {
    "system": """You are an expert in mathematical optimization and operations research.
Your goal is to design a globally consistent mathematical model and implement it in Gurobi Python to solve the given problem.
Your reasoning must inside <think> and </think> tags and follow the strict format:
<think>
<strategy>
1. **Paradigm Selection**: Identify the problem class (e.g., TSP, Scheduling, Facility Location) and choose **the most robust and efficient** modeling paradigm (e.g., Time-indexed, Flow-based, Big-M).
2. **Decision Variables**: Define the core variables (names, indices, categories) that serve as the foundation.
3. **Constraint Logic**: Identify key couplings. Explain how the defined variables will interact to enforce complex constraints.
</strategy>
<modeling>
Formulate the model following the design: Sets, Parameters, Variables, Objective, Constraints.
Ensure every segment must match the strategy.
</modeling>
<check>
1. Consistency Check: Does the model strictly follow the selected strategy?
2. Logic Check: Are the dependencies between variables and constraints logically sound?
3. Dimension Check: Do the variable indices in the code match the constraint loops?
4. Efficiency Check: Is the formulation compact and there are no redundant variables?
</check>
</think>
Output the final Python code inside a code block:
```python
# ... final code ...\n
""",
    "user": """Here is the problem:
{Question}
# Note:
- The Code must include:```python

import gurobipy as gp
from gurobipy import GRB
```
- Make sure the model variable is named `model`.
- Avoid using "<" and ">" in Gurobi constraints; instead, use "<=" or ">=" as appropriate.
- Carefully determine whether the variable is an integer or a continuous variable.
"""
}


def load_data(data_file: str) -> List[Dict]:
    samples = []
    with open(data_file, 'r', encoding='utf-8') as fd:
        for line in fd:
            example = json.loads(line)
            template = TEMPLATES

            question_key = f"en_question" if f"en_question" in example else "question"

            system_prompt = template["system"]
            user_prompt = template["user"].replace("{Question}", example[question_key]).strip()

            example_t = {k: v for k, v in example.items() if k not in ["prompt", "system_prompt", "user_prompt"]}
            example_t["system_prompt"] = system_prompt
            example_t["user_prompt"] = user_prompt
            example_t["prompt"] = f"{system_prompt}\n\n{user_prompt}"
            samples.append(example_t)

    print(f"Loaded {len(samples)} samples from '{data_file}'")

    for i, ex in enumerate(samples):
        if "id" not in ex:
            ex["id"] = i

    return samples


def init_model(args):
    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True
    )
    print("Model initialized.")
    return model


def get_sampling_params(args):
    stop_tokens = ["</s>"]
    if args.decoding_method == "greedy":
        return SamplingParams(
            n=args.topk,
            temperature=0,
            top_p=1,
            top_k=1,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            repetition_penalty=args.repetition_penalty,
            skip_special_tokens=False,
        )
    elif args.decoding_method == "sampling":
        return SamplingParams(
            n=args.topk,
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
            stop=stop_tokens,
            repetition_penalty=args.repetition_penalty,
            skip_special_tokens=False,
        )
    else:
        raise ValueError(f"Unsupported decoding method: {args.decoding_method}")


def extract_code_block(llm_output: str) -> str:
    """提取 ```python ... ``` 中的代码"""
    pattern = r'```python(.*?)```'
    match = re.search(pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def generate_and_save(model, samples, sampling_params, args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, os.path.basename(args.data_file).replace(".jsonl", "") + ".jsonl")

    system_prompts = [ex["system_prompt"] for ex in samples]
    user_prompts = [ex["user_prompt"] for ex in samples]
    prompts = [ex["prompt"] for ex in samples]
    ids = [ex["id"] for ex in samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    generations_k = [[] for _ in ids]

    for i in range(args.passk):
        texts = []
        for system_prompt, user_prompt in zip(system_prompts, user_prompts):
            if tokenizer and tokenizer.chat_template:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                texts.append(text)
            else:
                text = f"{system_prompt}\n\n{user_prompt}"
                texts.append(text)

        results = model.generate(texts, sampling_params)
        results = [g.outputs[0].text for g in results]

        for idx, content in enumerate(results):
            generations_k[idx].append(content)


    with open(save_path, "w", encoding="utf-8") as fw:
        for example, prompt, outputs in tqdm(zip(samples, prompts, generations_k), total=len(samples)):
            updated_example = example.copy()
            updated_example["prompt"] = prompt
            updated_example["raw_generations"] = outputs

            final_codes = []
            for raw_code_text in outputs:
                if isinstance(raw_code_text, (tuple, list)):
                    raw_code_text = raw_code_text[0]
                code_snippet = extract_code_block(raw_code_text)
                if not code_snippet:
                    code_snippet = raw_code_text
                final_codes.append(code_snippet)
            updated_example["en_gurobi_code"] = final_codes

            dump_str = json.dumps(updated_example, ensure_ascii=False)
            fw.write(dump_str + "\n")

    print(f"Saved {len(samples)} records to {save_path}")


def main(args):
    data_samples = load_data(args.data_file)
    model = init_model(args)
    sampling_params = get_sampling_params(args)
    generate_and_save(model, data_samples, sampling_params, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized evaluation script for language models")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input data file")

    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--topk", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument("--decoding_method", type=str, default="greedy",
                        choices=["greedy", "sampling"],
                        help="Decoding method")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")

    parser.add_argument("--passk", type=int, default=1, help="Number of top generations for pass@k")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save raw generation outputs for each sample. If not set, won't save them.")

    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    main(args)
