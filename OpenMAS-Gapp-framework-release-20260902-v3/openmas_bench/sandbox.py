"""Small, bounded execution sandboxes for Q2 dataset adapters.

The code sandbox executes only generated Python in a temporary directory with
an aggressive timeout. SWE-bench has no repository checkout in the compact
dataset snapshot, so its adapter validates patch structure and test-contract
references without claiming patch correctness.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any


def run_python_tests(dataset: str, row: dict[str, Any], generated: str, timeout: int = 8) -> dict[str, Any]:
    if not generated or len(generated) > 30000:
        return {"passed": False, "sandbox": "python_subprocess", "error": "empty_or_oversize_code"}
    raw = row.get("raw") or {}
    aliases = []
    if dataset == "HumanEval":
        prompt = raw.get("prompt") or row.get("question", "")
        tests = raw.get("test", "")
        entry = raw.get("entry_point", "")
        source = prompt + "\n" + generated + "\n" + tests + f"\ncheck({entry})\n"
    elif dataset == "MBPP":
        tests = "\n".join(raw.get("test_list") or [])
        # MBPP rows do not always put the required function name in the
        # natural-language prompt. Recover the public API from the provided
        # test calls (not from the gold implementation) and add a narrow alias
        # when the model used a different single function name.
        expected = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", tests)))
        try:
            tree = ast.parse(generated)
            defined = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        except SyntaxError:
            defined = []
        if len(defined) == 1 and len(expected) == 1 and defined[0] != expected[0]:
            aliases.append(f"{expected[0]} = {defined[0]}")
        source = generated + ("\n" + "\n".join(aliases) if aliases else "") + "\n" + tests + "\n"
    else:
        return {"passed": False, "sandbox": "unsupported"}
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"passed": False, "sandbox": "python_subprocess", "error": f"syntax:{exc.msg}"}
    with tempfile.TemporaryDirectory(prefix="openmas-code-") as td:
        path = Path(td) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=td,
                                  capture_output=True, text=True, timeout=timeout)
            return {"passed": proc.returncode == 0, "sandbox": "python_subprocess",
                    "returncode": proc.returncode, "stderr": proc.stderr[-1000:],
                    "interface_alias_applied": bool(aliases)}
        except subprocess.TimeoutExpired:
            return {"passed": False, "sandbox": "python_subprocess", "error": "timeout"}


def validate_swe_patch(row: dict[str, Any], generated: str) -> dict[str, Any]:
    raw = row.get("raw") or {}
    if not generated or "diff --git" not in generated:
        return {"passed": False, "sandbox": "patch_test_proxy", "error": "not_unified_diff"}
    targets = set(re.findall(r"^diff --git a/(.+?) b/(.+?)$", generated, re.MULTILINE))
    expected = str(raw.get("patch", ""))
    expected_paths = set(re.findall(r"^diff --git a/(.+?) b/(.+?)$", expected, re.MULTILINE))
    path_ok = not expected_paths or bool(targets & expected_paths)
    has_test_contract = bool(raw.get("FAIL_TO_PASS") or raw.get("test_patch"))
    return {"passed": bool(path_ok and has_test_contract), "sandbox": "patch_test_proxy",
            "path_match": path_ok, "test_contract_present": has_test_contract,
            "repository_checkout": False}


def run_swebench_tests(row: dict[str, Any], generated: str, timeout: int = 90) -> dict[str, Any]:
    """Apply a generated patch to a downloaded base snapshot and run tests.

    The compact benchmark bundle stores GitHub commit archives rather than full
    clones. This still gives a genuine patch-application and test execution
    signal while keeping each run isolated and bounded.
    """
    raw = row.get("raw") or {}
    commit = str(raw.get("base_commit", ""))
    if not commit or not generated or ("diff --git" not in generated and not re.search(r"^--- a/.+\n\+\+\+ b/", generated, re.MULTILINE)):
        return {"passed": False, "sandbox": "patch_test", "error": "missing_commit_or_diff", "repository_checkout": False}
    root = Path(__file__).resolve().parents[3] / "q2_datasets" / "swebench_repos"
    archives = list(root.glob(f"*/base-{commit}.tar.gz"))
    if not archives:
        return {"passed": False, "sandbox": "patch_test", "error": "base_archive_missing", "repository_checkout": False}
    fail_tests = _json_list(raw.get("FAIL_TO_PASS"))
    pass_tests = _json_list(raw.get("PASS_TO_PASS"))
    tests = fail_tests + pass_tests
    if not tests:
        return {"passed": False, "sandbox": "patch_test", "error": "test_contract_missing", "repository_checkout": True}
    with tempfile.TemporaryDirectory(prefix="openmas-swe-") as td:
        work = Path(td) / "repo"
        work.mkdir()
        try:
            subprocess.run(["tar", "-xzf", str(archives[0]), "--strip-components=1", "-C", str(work)], check=True, capture_output=True, timeout=60)
            patch_path = Path(td) / "candidate.patch"
            with patch_path.open("w", encoding="utf-8", newline="") as stream:
                stream.write(generated.replace("\r\n", "\n"))
            applied = subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=work, capture_output=True, text=True, timeout=30)
            if applied.returncode != 0:
                return {"passed": False, "sandbox": "patch_test", "error": "patch_not_applicable", "patch_stderr": applied.stderr[-1200:], "repository_checkout": True}
            subprocess.run(["git", "apply", str(patch_path)], cwd=work, check=True, capture_output=True, timeout=30)
            test_patch = str(raw.get("test_patch", ""))
            if test_patch:
                test_patch_path = Path(td) / "test.patch"
                with test_patch_path.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(test_patch.replace("\r\n", "\n"))
                test_apply = subprocess.run(["git", "apply", str(test_patch_path)], cwd=work, capture_output=True, text=True, timeout=30)
                if test_apply.returncode != 0:
                    return {"passed": False, "sandbox": "patch_test", "error": "test_patch_not_applicable", "patch_stderr": test_apply.stderr[-1200:], "repository_checkout": True}
            selected = tests
            env_python = root.parent / "swebench_py39" / "python.exe"
            python = str(env_python) if env_python.exists() else sys.executable
            # Astropy 4/5 has compiled extensions. Rebuilding it for every
            # candidate is prohibitively expensive, so run against the matching
            # installed wheel while overlaying only files changed by the patch.
            site = Path(td) / "site"
            probe = subprocess.run([python, "-c", "import astropy, pathlib; print(pathlib.Path(astropy.__file__).parent)"], capture_output=True, text=True, timeout=20)
            if probe.returncode != 0:
                return {"passed": False, "sandbox": "patch_test", "error": "environment_import_failed", "stderr": probe.stderr[-1200:], "repository_checkout": True}
            installed = Path(probe.stdout.strip())
            shutil.copytree(installed, site / "astropy")
            changed = {b for _, b in re.findall(r"^diff --git a/(.+?) b/(.+?)$", generated, re.MULTILINE)}
            if not changed:
                changed = set(re.findall(r"^\+\+\+ b/(.+?)$", generated, re.MULTILINE))
            for rel in changed:
                source = work / rel
                if source.exists() and rel.startswith("astropy/"):
                    target = site / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            env = dict(__import__("os").environ)
            env["PYTHONPATH"] = str(site)
            env["HOME"] = str(Path(td) / "home")
            env["USERPROFILE"] = env["HOME"]
            cert = subprocess.run([python, "-c", "import certifi; print(certifi.where())"], capture_output=True, text=True, timeout=20)
            if cert.returncode == 0:
                env["SSL_CERT_FILE"] = cert.stdout.strip()
                env["REQUESTS_CA_BUNDLE"] = cert.stdout.strip()
            detached = Path(td) / "tests"
            detached.mkdir()
            mapped: list[str] = []
            copied: dict[str, Path] = {}
            for test in selected:
                file_part, *node_parts = test.split("::")
                if file_part not in copied:
                    target = detached / Path(file_part).name
                    shutil.copy2(work / file_part, target)
                    data_dir = (work / file_part).parent / "data"
                    if data_dir.exists() and not (detached / "data").exists():
                        shutil.copytree(data_dir, detached / "data")
                    copied[file_part] = target
                mapped.append(str(copied[file_part]) + ("::" + "::".join(node_parts) if node_parts else ""))
            # Evaluate FAIL_TO_PASS and PASS_TO_PASS independently. A valid
            # repair must pass both contracts after the real git apply.
            fail_n = len(fail_tests)
            fail_mapped = mapped[:fail_n]
            pass_mapped = mapped[fail_n:]
            fail_proc = subprocess.run([python, "-m", "pytest", "-q", *fail_mapped], cwd=td, env=env, capture_output=True, text=True, timeout=timeout) if fail_mapped else None
            pass_proc = subprocess.run([python, "-m", "pytest", "-q", *pass_mapped], cwd=td, env=env, capture_output=True, text=True, timeout=timeout) if pass_mapped else None
            fail_ok = bool(fail_proc and fail_proc.returncode == 0)
            pass_ok = bool(pass_proc and pass_proc.returncode == 0)
            return {"passed": bool(fail_ok and pass_ok), "sandbox": "patch_test", "repository_checkout": True,
                    "patch_applied": True, "tests_run": selected,
                    "fail_to_pass_passed": fail_ok, "pass_to_pass_passed": pass_ok,
                    "fail_to_pass_count": len(fail_tests), "pass_to_pass_count": len(pass_tests),
                    "stderr": ((fail_proc.stderr if fail_proc else "") + (pass_proc.stderr if pass_proc else ""))[-1500:],
                    "stdout": ((fail_proc.stdout if fail_proc else "") + (pass_proc.stdout if pass_proc else ""))[-1500:]}
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
            return {"passed": False, "sandbox": "patch_test", "error": type(exc).__name__, "repository_checkout": True}


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []
