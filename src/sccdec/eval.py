import asyncio
import argparse
import os
import json
import re
import tempfile
from tqdm import tqdm
from openai import AsyncOpenAI
import httpx
import os.path
import subprocess


client = httpx.AsyncClient()

C_INCLUDE = """
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long ull;
"""

C_FUNC_EXAMPLE = """
bool func0(int num) {
    if (num <= 1) {
        return false;
    }
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}
""".strip()


async def compile_online(code, optimization_level, func_name="func0", compiler="cg141"):
    asm = await client.post(
        f"https://c.compiler-explorer.com/api/compiler/{compiler}/compile",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json={
            "source": code,
            "options": {
                "userArguments": f"-shared -fPIC -fcf-protection=full -{optimization_level.strip('-')}",
                "compilerOptions": {"skipAsm": False, "executorRequest": False},
                "filters": {
                    "binary": True,
                    "binaryObject": False,
                    "commentOnly": True,
                    "demangle": True,
                    "directives": True,
                    "execute": False,
                    "intel": False,
                    "labels": False,
                    "libraryCode": False,
                    "trim": True,
                    "debugCalls": False,
                },
                "tools": [],
                "libraries": [],
            },
            "lang": "c",
            "allowStoreCodeDebug": True,
        },
        timeout=60,
    )
    result = asm.json()
    if len(result["asm"]) == 1 and result["asm"][0]["text"] == "<Compilation failed>":
        raise Exception("<Compilation failed>")

    # from pprint import pprint
    # pprint(result)
    asm_codes = []
    for item in result["asm"]:
        asm_code: str = item["text"]
        if asm_code.startswith(func_name):
            asm_codes.append(f"<{func_name}>:")
        elif asm_code.startswith(" ") and len(asm_codes):
            # remove comments
            asm_code = asm_code.rsplit("#", maxsplit=1)[0].strip()
            asm_codes.append(asm_code.strip())
        elif len(asm_codes) and not asm_code.startswith(" "):
            break

    return "\n".join(asm_codes)


async def compile(code, optimization_level, func_name="func0"):
    with tempfile.TemporaryDirectory() as tmpdir_path:
        code_file = os.path.join(tmpdir_path, "code.c")
        binary_file = os.path.join(tmpdir_path, "code.out")
        optimization_level = optimization_level.strip("-")

        with open(code_file, "w") as f:
            f.write(code)

        # Compile the C program to get the binary
        proc = await asyncio.create_subprocess_exec(
            "gcc",
            "-shared",
            "-fPIC",
            f"-{optimization_level}",
            "-lm",
            "-o",
            binary_file,
            code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the compiler...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing the compiler...")
                proc.kill()
                await proc.wait()
        except Exception as e:
            print(e)

        if not os.path.exists(binary_file):
            return None

        # Disassemble the binary to get the assembly code
        proc = await asyncio.create_subprocess_exec(
            "objdump", "-d", binary_file, stdout=subprocess.PIPE
        )
        try:
            asm, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the objdump...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing the objdump...")
                proc.kill()
                await proc.wait()
            return None
        except Exception:
            return None
        asm = asm.decode()
        # IMPORTANT replace func0 with the function name
        if "<" + func_name + ">:" not in asm:
            raise ValueError("compile fails")
        # IMPORTANT replace func0 with the function name
        asm = (
            "<"
            + func_name
            + ">:"
            + asm.split("<" + func_name + ">:")[-1].split("\n\n")[0]
        )
        asm_clean = ""
        asm_sp = asm.split("\n")
        for tmp in asm_sp:
            if len(tmp.split("\t")) < 3 and "00" in tmp:
                continue
            idx = min(len(tmp.split("\t")) - 1, 2)
            tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
            tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
            asm_clean += tmp_asm + "\n"
        return asm_clean.strip()


async def evaluate_func(c_func, c_test, c_func_decompile):
    with tempfile.TemporaryDirectory() as tempdir:
        c_include = C_INCLUDE
        for line in c_func.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_func = c_func.replace(line, "")
        for line in c_test.split("\n"):
            if "#include" in line:
                c_include += line + "\n"
                c_test = c_test.replace(line, "")
        c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test

        # Define the C file and executable names
        c_file = os.path.join(tempdir, "combine.c")
        executable = os.path.join(tempdir, "combine")
        if os.path.exists(executable):
            os.remove(executable)

        with open(c_file, "w") as f:
            f.write(c_combine)

        # Compile the C program to an executable
        try:
            proc = await asyncio.create_subprocess_exec(
                "gcc",
                c_file,
                "-o",
                executable,
                "-lm",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the compiler...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing the compiler...")
                proc.kill()
                await proc.wait()
        except Exception:
            pass

        if not os.path.exists(executable):
            return 0, 0, None

        # Run the compiled executable
        proc = await asyncio.create_subprocess_exec(
            executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            flag_run = int(0 == await asyncio.wait_for(proc.wait(), timeout=10))
        except asyncio.TimeoutError:
            print("Timeout reached, terminating the process...")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                print("Process did not terminate, killing it...")
                proc.kill()
                await proc.wait()
            flag_run = 0
        except Exception:
            flag_run = 0

        return 1, flag_run, None


async def model_decompile(
    model="model",
    timeout=9999,
    max_tokens=1024,
    temperature=0.01,
    stream=False,
    prompt=None,
    messages=None,
    client: AsyncOpenAI = None,
    **chat_params,
):
    assert client is not None, "client is required"

    if messages is None:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    chat_completion_resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        **chat_params,
    )
    content = chat_completion_resp.choices[0].message.content
    try:
        # regex find last ```c...```
        codes = re.findall(r"```\w*\n(?P<code>.*?)```", content, re.DOTALL)
        return codes[-1] if codes else content
    except Exception as e:
        return content

    return content


async def build_icl_example(
    c_func, optimization_level, c_func_decompile, compiler=None
):
    c_include = ""
    for line in c_func.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_func = c_func.replace(line, "")
    c_onlyfunc = c_include + "\n" + c_func_decompile

    if compiler is None:
        asm = await compile(code=c_onlyfunc, optimization_level=optimization_level)
    else:
        asm = await compile_online(
            code=c_onlyfunc, optimization_level=optimization_level, compiler=compiler
        )
    if asm is None:
        return None

    c_func_decompile = c_func_decompile.strip()
    return [
        {
            "role": "user",
            "content": f"# This is the assembly code:\n{asm.strip()}\n# What is the source code?",
        },
        {
            "role": "assistant",
            "content": c_func_decompile,
        },
    ]


async def run_decompile(
    sem,
    item,
    client: AsyncOpenAI = None,
    model="model",
    re_opt_state=None,
    compiler=None,
    one_shot=False,
    context=None,
):
    async with sem:
        c_func = item["c_func"]
        c_test = item["c_test"]
        input_asm_prompt = item["input_asm_prompt"]
        opt_state = item["type"]

        if one_shot:
            messages = [
                {
                    "role": "user",
                    "content": f"# This is the assembly code:\n{context[opt_state]}\n# What is the source code?",
                },
                {
                    "role": "assistant",
                    "content": context["func"],
                },
                {
                    "role": "user",
                    "content": f"# This is the assembly code:\n{input_asm_prompt.strip()}\n# What is the source code?",
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": f"# This is the assembly code:\n{input_asm_prompt.strip()}\n# What is the source code?",
                },
            ]

        if not re_opt_state:
            re_opt_state = opt_state

        try:
            if item.get("c_func_decompile", None) is not None:
                c_func_decompile, flag_compile, flag_run = (
                    item["c_func_decompile"],
                    item["compile"],
                    item["run"],
                )
            else:
                c_func_decompile = await model_decompile(
                    client=client, model=model, messages=messages, temperature=0.0
                )
                flag_compile, flag_run, err_info = await evaluate_func(
                    c_func, c_test, c_func_decompile
                )
            try:
                if item.get("c_func_re_decompile", None) is not None:
                    c_func_re_decompile, flag_re_compile, flag_re_run = (
                        item["c_func_re_decompile"],
                        item["re_compile"],
                        item["re_run"],
                    )
                elif flag_compile == 1:
                    scc_context = await build_icl_example(
                        c_func, re_opt_state, c_func_decompile, compiler=compiler
                    )
                    c_func_re_decompile = await model_decompile(
                        client=client,
                        model=model,
                        messages=scc_context + messages[-1:],
                        temperature=0.0,
                    )
                    flag_re_compile, flag_re_run, err_info = await evaluate_func(
                        c_func, c_test, c_func_re_decompile
                    )
                else:
                    flag_re_compile, flag_re_run, c_func_re_decompile = (
                        flag_compile,
                        flag_run,
                        None,
                    )
            except Exception as e1:
                flag_re_compile, flag_re_run, c_func_re_decompile = (
                    flag_compile,
                    flag_run,
                    None,
                )
                pass

        except Exception as e2:
            return {
                "opt_state": opt_state,
                "compile": 0,
                "run": 0,
                "re_compile": 0,
                "re_run": 0,
                "c_func": item["c_func"],
                "c_func_decompile": None,
                "c_func_re_decompile": None,
            }

        return {
            "opt_state": opt_state,
            "run": flag_run,
            "compile": flag_compile,
            "re_run": flag_re_run,
            "re_compile": flag_re_compile,
            "c_func": item["c_func"],
            "c_func_decompile": c_func_decompile,
            "c_func_re_decompile": c_func_re_decompile,
        }


async def eval_model(
    client,
    data_all,
    num_semaphore=16,
    model_name="model",
    model_tag="model",
    re_opt_state=None,
    output_file="result.jsonl",
    result_file=None,
    compiler=None,
    one_shot=False,
):
    context = {
        "func": C_FUNC_EXAMPLE,
    }

    if one_shot:
        for opt_state in ["O0", "O1", "O2", "O3"]:
            asm = await compile(
                code=C_INCLUDE + C_FUNC_EXAMPLE,
                optimization_level=opt_state,
                func_name="func0",
            )
            context[opt_state] = asm

    semaphore = asyncio.Semaphore(num_semaphore)
    num_re_compile = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
    num_re_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
    num_compile = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}
    num_run = {"O0": 0, "O1": 0, "O2": 0, "O3": 0}

    num_result = {
        "O0": 0,
        "R0": 0,
        "O1": 0,
        "R1": 0,
        "O2": 0,
        "R2": 0,
        "O3": 0,
        "R3": 0,
    }
    tasks = [
        asyncio.create_task(
            run_decompile(
                semaphore,
                item,
                client=client,
                model=model_name,
                re_opt_state=re_opt_state,
                compiler=compiler,
                one_shot=one_shot,
                context=context,
            )
        )
        for item in data_all
    ]

    # Use tqdm to create a progress bar for the asyncio.gather
    results = []
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model_tag)
    for f in pbar:
        try:
            result = await f
            results.append(result)

            opt_state = result["opt_state"]
            num_compile[opt_state] += result["compile"]
            num_run[opt_state] += result["run"]

            num_re_compile[opt_state] += result["re_compile"]
            num_re_run[opt_state] += result["re_run"]

            num_result[opt_state] += result["run"]
            num_result[f"R{opt_state[-1]}"] += result["re_run"]
        except Exception as e:
            raise e

        pbar.set_postfix(num_result)

    with open(output_file, "a") as f:
        total_run = sum(num_run.values())
        total_re_run = sum(num_re_run.values())
        total_run_rate = total_run / len(data_all)
        total_re_run_rate = total_re_run / len(data_all)

        total_compile = sum(num_compile.values())
        total_re_compile = sum(num_re_compile.values())
        total_compile_rate = total_compile / len(data_all)
        total_re_compile_rate = total_re_compile / len(data_all)

        level_num = len(data_all) // 4

        data = {
            "model": model_tag,
            # rates
            "total_run_rate": total_run_rate,
            "total_compile_rate": total_compile_rate,
            "run_rate": {k: v / level_num for k, v in num_run.items()},
            "compile_rate": {k: v / level_num for k, v in num_compile.items()},
            "total_re_run_rate": total_re_run_rate,
            "total_re_compile_rate": total_re_compile_rate,
            "re_run_rate": {k: v / level_num for k, v in num_re_run.items()},
            "re_compile_rate": {k: v / level_num for k, v in num_re_compile.items()},
            # numbers
            "total_run": total_run,
            "total_re_run": total_re_run,
            "total_compile": total_compile,
            "total_re_compile": total_re_compile,
            "num_run": num_run,
            "num_compile": num_compile,
            "num_re_run": num_re_run,
            "num_re_compile": num_re_compile,
        }

        f.write(json.dumps(data) + "\n")

    if result_file:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/decompile-eval-gcc.json",
    )
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str, default="None")
    parser.add_argument("--model_name", type=str, default="model")

    parser.add_argument("--num_semaphore", default=16)
    parser.add_argument("--compiler", default=None, required=False)
    parser.add_argument("--one_shot", default=False, action="store_true")
    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    with open(args.data_path, "r") as f:
        data_all = json.load(f)

    model_name = args.model_name.strip("/").replace("/", "_")
    result_file = os.path.join("results", f"{model_name}.json")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if not os.path.exists(result_file):
        await eval_model(
            client=client,
            data_all=data_all,
            model_name=args.model_name,
            model_tag=model_name + "[1shot]" if args.one_shot else model_name,
            output_file=os.path.join("results", "resultss.jsonl"),
            result_file=result_file,
            one_shot=args.one_shot,
            compiler=args.compiler,
        )


if __name__ == "__main__":
    asyncio.run(main())
