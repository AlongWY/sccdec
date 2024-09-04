from asyncio import subprocess
import argparse
import json
import os
import asyncio
import re
import tempfile
from tqdm import tqdm

os.environ["USE_TORCH"] = "FALSE"
from datasets import load_from_disk


async def compile_with_optimization(
    tmpdir, formatted_code_file, function_name, optimization
):
    binary_file = os.path.join(tmpdir, f"code{optimization}.out")
    # Compile the C program to get the binary
    try:
        proc_compile = await asyncio.create_subprocess_exec(
            "gcc",
            "-shared",
            "-fPIC",
            "-g3",
            optimization,
            "-o",
            binary_file,
            formatted_code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.wait_for(proc_compile.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the process...")
        proc_compile.terminate()
        try:
            await asyncio.wait_for(proc_compile.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing it...")
            proc_compile.kill()
            await proc_compile.wait()

    if not os.path.exists(binary_file):
        raise ValueError("Binary not found!")

    # Decompile the binary to get the assembly code
    proc_decompile = await asyncio.create_subprocess_exec(
        "objdump",
        "-d",
        "-S",
        "--source-comment=;",
        binary_file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    asm, _ = await proc_decompile.communicate()
    asm = asm.decode("utf-8")

    function_name_chk = f"<{function_name}>:"
    # IMPORTANT replace func0 with the function name
    if function_name_chk not in asm:
        raise ValueError("Function not found in asm!")

    # IMPORTANT replace func0 with the function name
    asm = function_name_chk + asm.split(function_name_chk)[-1].split("\n\n")[0]
    asm_sp = asm.split("\n")

    code_blocks = []
    asm_blocks = []

    last_code = False
    for tmp in asm_sp:
        if tmp.startswith(";"):
            if last_code:
                code_blocks[-1] += tmp + "\n"
            else:
                code_blocks.append(tmp + "\n")
            last_code = True
            continue

        if len(tmp.split("\t")) < 3 and "00" in tmp:
            continue
        idx = min(len(tmp.split("\t")) - 1, 2)
        tmp_asm = "\t".join(tmp.split("\t")[idx:])  # remove the binary code
        tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments

        if last_code:
            asm_blocks.append(tmp_asm + "\n")
        elif not code_blocks:
            asm_blocks.append(tmp_asm + "\n")
        else:
            asm_blocks[-1] += tmp_asm + "\n"
        last_code = False

    # First code block keep only the function definition line
    first_block = code_blocks[0].split("\n")
    func_line_idx = 0
    for idx, line in enumerate(first_block):
        if function_name in line:
            func_line_idx = idx
            break
    code_blocks[0] = "\n".join(first_block[func_line_idx:]).strip() + "\n"

    # Reverse the data
    asm_clean = asm_blocks[0]
    assert len(code_blocks) + 1 == len(
        asm_blocks
    ), f"code: {len(code_blocks)} asm:{len(asm_blocks)}"
    for code_block, asm_block in zip(code_blocks, asm_blocks[1:]):
        asm_clean += asm_block + code_block

    return optimization, asm_clean


async def compile(idx, synth_deps, function_def, function_name):
    # remove static and inline before {
    function_def, remain = function_def.split("{", maxsplit=1)
    function_def = (
        function_def.replace("static", "")
        .replace("inline", "")
        .replace("\n", " ")
        .strip()
    )
    remain, right_bracket = remain.rsplit("}", maxsplit=1)
    # remain comments such as # 1 "filename.c"
    remain = re.sub(r"#\s+\d+\s+\"[^\"]+\"", "", remain)
    function_def += " {" + remain + "\n}"

    # replace multiple \n to one \n
    function_def = re.sub("\n+", "\n", function_def)

    # the stmt can't cross lines
    if synth_deps is not None:
        full_code = synth_deps.strip() + "\n" + function_def.strip() + "\n\n"
    else:
        full_code = function_def.strip() + "\n\n"

    with tempfile.TemporaryDirectory(dir="/run/user/10000") as tmpdir:
        code_file = os.path.join(tmpdir, "code.c")
        formatted_code_file = os.path.join(tmpdir, "formatted_code.c")
        with open(code_file, "w") as f:
            f.write(full_code)

        proc_compile = await asyncio.create_subprocess_exec(
            "clang-format",
            code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        formatted_code, _ = await proc_compile.communicate()
        formatted_code = formatted_code.decode("utf-8")

        with open(formatted_code_file, "w") as f:
            f.write(formatted_code)

        tasks = [
            asyncio.create_task(
                compile_with_optimization(
                    tmpdir, formatted_code_file, function_name, optimization
                )
            )
            for optimization in ["-O0", "-O1", "-O2", "-O3"]
        ]

        asm_codes = {}
        for example in asyncio.as_completed(tasks):
            try:
                optimization, asm = await example
                if asm is not None:
                    asm_codes[optimization] = asm
            except Exception as e:
                pass

        if len(asm_codes) == 0:
            raise ValueError("No asm code generated!")

        asm_codes["idx"] = idx
        asm_codes["synth_deps"] = synth_deps
        asm_codes["function_def"] = function_def
        asm_codes["function_name"] = function_name
        return asm_codes


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=30000, required=False)
    parser.add_argument("--data_dir", type=str, default="exebench", required=False)
    parser.add_argument(
        "--split", type=str, default="train_real_compilable", required=False
    )
    parser.add_argument("--output", type=str, default="data/data.jsonl", required=False)
    args = parser.parse_args()

    # check file exists and skip existing files
    idx_bias = -1
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in tqdm(f):
                try:
                    example = json.loads(line)
                    idx_bias = max(idx_bias, example["idx"])
                except json.JSONDecodeError:
                    print(f"Error: {line}")
                    exit(1)

    dataset = load_from_disk(os.path.join(args.data_dir, args.split))
    dataset = dataset.select(range(idx_bias + 1, len(dataset)), keep_in_memory=True)
    print(f"Remaining examples: {len(dataset)}")
    max_examples = min(args.num, len(dataset))
    with open(args.output, "a") as f:
        pbar = tqdm(enumerate(dataset), total=max_examples)
        for idx, item in pbar:
            if idx >= max_examples:
                break
            try:
                code_asm = await compile(
                    idx=idx_bias + 1 + idx,
                    synth_deps=item["synth_deps"],
                    function_def=item["func_def"],
                    function_name=item["fname"],
                )
                if code_asm is not None:
                    f.write(json.dumps(code_asm) + "\n")
            except Exception as e:
                pbar.set_postfix_str(f"Error[{idx_bias + 1 + idx}]: {e}")


if __name__ == "__main__":
    asyncio.run(main())
