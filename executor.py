import os
import io
import re
import copy
import pickle
import traceback
import multiprocessing as mp
from queue import Empty
from contextlib import redirect_stdout, redirect_stderr

from verl.reward_func_new.content_utils import extract_obj, extract_sol


class GenericRuntime:
    GLOBAL_DICT = {}    # Global variables available in the runtime
    LOCAL_DICT = None   # Local variables available in the runtime, can be None if not used
    HEADERS = []        # List of code snippets to execute at initialization

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece):
        if re.search(r"(\s|^)?input\(", code_piece) or re.search(r"(\s|^)?os.system\(", code_piece):
            raise RuntimeError("Input or system commands are not allowed")
        exec(code_piece, self._global_vars)

    def eval_code(self, expr):
        return eval(expr, self._global_vars)

    def inject(self, var_dict):
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars.get("answer")


class PythonExecutor:
    def __init__(self, runtime=None, timeout_length=5):
        self.runtime = runtime if runtime else GenericRuntime()
        self.timeout_length = timeout_length
        self._mp_ctx = mp.get_context("spawn")

    def process_generation_to_code(self, gens):
        return [g.split("\n") if g is not None else None for g in gens]

    @staticmethod
    def execute(code, runtime=None, runtime_global_vars=None):
        if runtime_global_vars is not None:
            runtime = GenericRuntime()
            runtime._global_vars.update(runtime_global_vars)
        elif runtime is None:
            runtime = GenericRuntime()
        
        try:
            program_io = io.StringIO()
            error_io = io.StringIO()
            with redirect_stdout(program_io), redirect_stderr(error_io):
                runtime.exec_code("\n".join(code))
            log_text = program_io.getvalue() + "\n" + error_io.getvalue()
            program_io.seek(0)
            error_io.seek(0)
            result = program_io.read()
            if result == "":
                runtime.exec_code(code[:-1])
                result = runtime.eval_code(code[-1])
            report = "Done"
            str(result)
            if result is not None:
                pickle.dumps(result)  # Serialization check
            return result, report, log_text
        except Exception as e:
            error_trace = traceback.format_exc()
            result = ""
            report = error_trace.split("\n")[-2] if error_trace else "Execution Error"
            log_text = error_trace
            return result, report, log_text

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    @staticmethod
    def run_code_in_subprocess(code, runtime_global_vars, queue):
        """
        Execute code inside an isolated subprocess.
        Result is sent back through the provided queue.
        """
        result, report, log_text = PythonExecutor.execute(code, runtime_global_vars=runtime_global_vars)
        queue.put((result, report, log_text))

    def collect_results_from_children(self, running_procs, results_dict):
        """Join running subprocesses with timeout and harvest results."""
        for idx, proc, queue in running_procs:
            proc.join(self.timeout_length)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                results_dict[idx] = ("", "Timeout Error", "")
            else:
                try:
                    results_dict[idx] = queue.get_nowait()
                except Empty:
                    results_dict[idx] = ("", "Execution Error", "")
            queue.close()
            queue.join_thread()

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)
        all_exec_results = []

        max_workers = max(1, min(len(all_code_snippets), os.cpu_count() // 4))
        runtime_global_vars = self.runtime._global_vars.copy() if self.runtime else {}
        results_dict = {}
        running = []

        for idx, code_snippet in enumerate(all_code_snippets):
            queue = self._mp_ctx.Queue()
            proc = self._mp_ctx.Process(
                target=self.run_code_in_subprocess,
                args=(code_snippet, runtime_global_vars, queue)
            )
            proc.start()
            running.append((idx, proc, queue))
            if len(running) >= max_workers:
                self.collect_results_from_children(running, results_dict)
                running = []

        if running:
            self.collect_results_from_children(running, results_dict)

        for i in range(len(all_code_snippets)):
            all_exec_results.append(results_dict.get(i, ("", "Execution Error", "")))
        
        batch_obj = []
        batch_sol = []
        batch_report = []
        batch_logs = []
        for code, (res, report, log_text) in zip(all_code_snippets, all_exec_results):
            res, report = str(res).strip(), str(report).strip()
            batch_obj.append(extract_obj(res))
            sol = extract_sol(res)
            batch_sol.append(sol)
            batch_report.append(report)
            batch_logs.append(log_text)
        
        return batch_obj, batch_sol, batch_report, batch_logs