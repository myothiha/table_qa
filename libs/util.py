import re
import multiprocessing

import subprocess
import tempfile
import signal

def extract_function4(data):
    """
    Extracts any top-level import lines and the first `answer()` function,
    cleaned and truncated to the last `return` statement.
    """
    # Collect top-level import statements (before answer function)
    import_lines = "\n".join(re.findall(r"^\s*(import [^\n]+|from [^\n]+ import [^\n]+)", data, re.MULTILINE))

    # Match the `answer()` function including up to its last return
    func_pattern = r"(def answer\(.*?\):(?:.|\n)*?return[^\n]*)(?=\n[^ \t]|$)"
    match = re.search(func_pattern, data, re.DOTALL)
    print("Func match", match)

    if not match:
        return "No valid answer() function found."

    function_body = match.group(0).strip()

    # Trim to the last return
    last_return_match = re.search(r"(.*return[^\n]*)(?=$|\n)", function_body, re.DOTALL)
    if last_return_match:
        function_body = function_body[:function_body.rfind(last_return_match.group(0)) + len(last_return_match.group(0))].strip()

    return f"{import_lines}\n\n{function_body}".strip()


# def extract_function4(data):
#     """
#     Extract the first `answer()` function and include everything up to the last `return` statement.

#     Parameters:
#         data (str): The input string containing multiple functions.

#     Returns:
#         str: The extracted `answer()` function, cleaned and truncated to the last `return` statement.
#     """
#     # Regex to capture the first `answer()` function
#     # pattern = r"(def answer\(.*?\):.*?return.*?)(?=def|$)"
#     pattern = r"(def answer\(.*?\):(?:\n(?: {4}|\t).*)*?\n( {4}|\t)return[^\n]*)(?=\n[^ \t]|$)"
#     match = re.search(pattern, data, re.DOTALL)

#     if match:
#         # Extract the matched function
#         function_body = match.group(0).strip()
        
#         # Find the last `return` statement
#         last_return_match = re.search(r"(.*return[^\n]*)(?=$|\n)", function_body, re.DOTALL)
#         if last_return_match:
#             # Include everything up to and including the last `return` statement
#             return function_body[:function_body.rfind(last_return_match.group(0)) + len(last_return_match.group(0))].strip()
        
#         # If no return found, return the full function
#         return function_body.strip().replace("[end of text]", "")
#     else:
#         return "No valid `answer()` function found."
    

def _exec_code(code_str, df, queue):
    try:
        local_vars = {"df": df, "pd": __import__("pandas")}
        exec(code_str, local_vars, local_vars)
        queue.put(local_vars.get("ans", "__NO_ANSWER__"))
    except Exception as e:
        queue.put(f"__CODE_ERROR__: {e}")

def postprocess4(response: str, df, timeout_sec=4):
    try:
        code = extract_function4(response)
        code += "\nans = answer(df)"
    except Exception as e:
        return f"__CODE_ERROR__: {e}"

    # print("=== CODE TO EXECUTE ===")
    # print(code)
    # print("=== END OF CODE ===")

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_exec_code, args=(code, df, queue))
    process.start()
    process.join(timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join()
        return "__CODE_ERROR__: Code took too long to run."

    try:
        result = queue.get(timeout=1)
    except Exception as e:
        return f"__CODE_ERROR__: Queue_Error: {e}"

    return str(result).split("\n")[0] if "\n" in str(result) else result

def run_code_simple(response, df):
    """
    Safely executes the given code string that defines an `answer(df)` function,
    then runs it and returns the result or error.
    """

    try:
        code = extract_function4(response)
        code += "\nans = answer(df)"
    except Exception as e:
        return f"__CODE_ERROR__: {e}"
    
    try:
        local_vars = {"df": df, "pd": __import__("pandas")}
        exec(code + "\nans = answer(df)", local_vars)
        return local_vars.get("ans", "__NO_ANSWER__")
    except Exception as e:
        return f"__CODE_ERROR__: {e}"

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

def run_code_with_timeout(code_str, df, timeout_sec=3):
    print(code_str)
    # code_str = extract_function4(response)
    def safe_exec():
        local_vars = {"df": df, "pd": __import__("pandas")}
        exec(code_str + "\nans = answer(df)", local_vars)
        return local_vars.get("ans", "__NO_ANSWER__")

    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    try:
        result = safe_exec()
        signal.alarm(0)  # Cancel the alarm if execution succeeds
        return result

    except TimeoutException as e:
        return f"__CODE_ERROR__: Timeout - {e}"

    except Exception as e:
        return f"__CODE_ERROR__: {e}"

    finally:
        signal.alarm(0)  # Always clear the alarm in case of error



def run_code_subprocess(code: str, df, timeout=4):
    import pandas as pd
    import pickle

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the DataFrame to a file
        df_path = f"{tmpdir}/df.pkl"
        pickle.dump(df, open(df_path, "wb"))

        # Save code to a file
        code_path = f"{tmpdir}/code.py"
        result_path = f"{tmpdir}/result.txt"
        with open(code_path, "w") as f:
            f.write(f"""
import pandas as pd
import pickle
df = pickle.load(open("{df_path}", "rb"))
{code}
with open("{result_path}", "w") as out:
    out.write(str(ans))
""")

        try:
            subprocess.run(["python", code_path], timeout=timeout, check=True)
            with open(result_path, "r") as f:
                return f.read().strip()
        except subprocess.TimeoutExpired:
            return "__CODE_ERROR__: Code took too long to run."
        except subprocess.CalledProcessError as e:
            return f"__CODE_ERROR__: {e}"
        except Exception as e:
            return f"__CODE_ERROR__: {e}"
