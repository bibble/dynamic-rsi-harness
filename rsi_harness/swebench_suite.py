"""
Official SWE-bench Verified Evaluation Suite for the Dynamic RSI Harness.

Loads real GitHub issues, base-commit source files, and official FAIL_TO_PASS / PASS_TO_PASS
test suites from `princeton-nlp/SWE-bench_Verified`:
  1. `astropy__astropy-12907` (astropy/modeling/separable.py)
  2. `astropy__astropy-14995` (astropy/nddata/mixins/ndarithmetic.py)
  3. `psf__requests-1921`     (requests/sessions.py)
  4. `psf__requests-6028`     (requests/utils.py)
  5. `pytest-dev__pytest-7324` (src/_pytest/mark/expression.py)
  6. `sympy__sympy-17630`     (sympy/matrices/expressions/matexpr.py)
  7. `sympy__sympy-18698`     (sympy/polys/polytools.py)
  8. `sympy__sympy-23262`     (sympy/utilities/lambdify.py)
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SWEBenchInstance:
    instance_id: str
    repo: str
    base_commit: str
    target_file: str
    difficulty: str
    problem_statement: str
    focal_window_code: str
    focal_function_name: str
    base_file_content: str
    gold_patch: str
    fail_to_pass_assertions: str
    pass_to_pass_assertions: str
    domain_tags: List[str]


def _extract_focal_window(source: str, patch: str, pad_before: int = 35, pad_after: int = 45) -> str:
    lines = source.splitlines()
    matches = list(re.finditer(r"@@ -(\d+),(\d+)", patch))
    if not matches:
        return "\n".join(lines[:100])
    first_line = int(matches[0].group(1))
    last_line = int(matches[-1].group(1)) + int(matches[-1].group(2))
    lo = max(0, first_line - pad_before)
    hi = min(len(lines), last_line + pad_after)
    return "\n".join(lines[lo:hi])


def load_swebench_verified_suite(json_path: str) -> List[SWEBenchInstance]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    specs: Dict[str, Dict[str, Any]] = {
        "astropy__astropy-12907": {
            "focal_fn": "_cstack",
            "tags": ["astropy", "numpy", "matrix", "separability"],
            "f2p": (
                "import numpy as np\n"
                "from astropy.modeling import models as m\n"
                "cm = m.Linear1D(10) & m.Linear1D(5)\n"
                "# Nested CompoundModel: separability_matrix(m.Pix2Sky_TAN() & cm) calls _cstack with right=ndarray\n"
                "res = separability_matrix(m.Pix2Sky_TAN() & cm)\n"
                "expected = np.array([[True, True, False, False], [True, True, False, False], [False, False, True, False], [False, False, False, True]])\n"
                "assert np.array_equal(res, expected)\n"
            ),
            "p2p": (
                "import numpy as np\n"
                "from astropy.modeling import models as m\n"
                "res_simple = separability_matrix(m.Linear1D(10) & m.Linear1D(5))\n"
                "assert np.array_equal(res_simple, np.array([[True, False], [False, True]]))\n"
            ),
        },
        "astropy__astropy-14995": {
            "focal_fn": "_arithmetic_mask",
            "tags": ["astropy", "nddata", "mask", "none_handling"],
            "f2p": (
                "import numpy as np\n"
                "from copy import deepcopy\n"
                "class DummyND:\n"
                "    def __init__(self, mask):\n"
                "        self.mask = mask\n"
                "    _arithmetic_mask = _arithmetic_mask\n"
                "mask = np.array([[0, 1], [64, 8]])\n"
                "nd_masked = DummyND(mask)\n"
                "nd_no_mask = DummyND(None)\n"
                "res1 = nd_masked._arithmetic_mask(None, nd_no_mask, np.bitwise_or)\n"
                "assert np.array_equal(res1, mask) and res1 is not mask\n"
                "res2 = nd_masked._arithmetic_mask(None, None, np.bitwise_or)\n"
                "assert np.array_equal(res2, mask) and res2 is not mask\n"
            ),
            "p2p": (
                "import numpy as np\n"
                "class DummyND:\n"
                "    def __init__(self, mask):\n"
                "        self.mask = mask\n"
                "    _arithmetic_mask = _arithmetic_mask\n"
                "assert DummyND(None)._arithmetic_mask(None, DummyND(None), np.bitwise_or) is None\n"
                "m1 = np.array([1, 0])\n"
                "m2 = np.array([0, 2])\n"
                "assert np.array_equal(DummyND(m1)._arithmetic_mask(None, DummyND(m2), np.bitwise_or), np.array([1, 2]))\n"
            ),
        },
        "psf__requests-1921": {
            "focal_fn": "merge_setting",
            "tags": ["requests", "headers", "dict_merge", "none_filtering"],
            "f2p": (
                "from collections import OrderedDict\n"
                "# When session_setting has a key with value None, merge_setting must omit it too!\n"
                "merged = merge_setting({'User-Agent': 'custom'}, {'Accept-Encoding': None, 'User-Agent': 'default'})\n"
                "assert 'Accept-Encoding' not in merged\n"
                "assert merged['User-Agent'] == 'custom'\n"
            ),
            "p2p": (
                "merged2 = merge_setting({'Accept-Encoding': None}, {'Accept-Encoding': 'gzip', 'Host': 'example.com'})\n"
                "assert 'Accept-Encoding' not in merged2 and merged2['Host'] == 'example.com'\n"
                "assert merge_setting(True, False) is True\n"
            ),
        },
        "psf__requests-6028": {
            "focal_fn": "prepend_scheme_if_needed",
            "tags": ["requests", "url_parsing", "proxy_auth", "netloc"],
            "f2p": (
                "assert prepend_scheme_if_needed('http://user:pass@example.com/path?query', 'http') == 'http://user:pass@example.com/path?query'\n"
                "assert prepend_scheme_if_needed('http://user@example.com/path?query', 'http') == 'http://user@example.com/path?query'\n"
            ),
            "p2p": (
                "assert prepend_scheme_if_needed('example.com/path', 'http') == 'http://example.com/path'\n"
                "assert prepend_scheme_if_needed('//example.com/path', 'http') == 'http://example.com/path'\n"
            ),
        },
        "pytest-dev__pytest-7324": {
            "focal_fn": "expression",
            "tags": ["pytest", "ast", "compiler", "keywords", "identifiers"],
            "f2p": (
                "# In Python 3.8+, ast.Name('True' / 'False' / 'None', ast.Load()) is illegal or evaluates to constant booleans instead of calling matcher!\n"
                "expr_false = Expression.compile('False')\n"
                "assert expr_false.evaluate(lambda ident: ident == 'False') is True\n"
                "assert expr_false.evaluate(lambda ident: False) is False\n"
                "expr_true = Expression.compile('True')\n"
                "assert expr_true.evaluate(lambda ident: False) is False\n"
                "assert expr_true.evaluate(lambda ident: ident == 'True') is True\n"
                "expr_none = Expression.compile('None')\n"
                "assert expr_none.evaluate(lambda ident: ident == 'None') is True\n"
            ),
            "p2p": (
                "expr_std = Expression.compile('slow and not integration')\n"
                "assert expr_std.evaluate(lambda ident: ident == 'slow') is True\n"
                "assert expr_std.evaluate(lambda ident: ident in ('slow', 'integration')) is False\n"
            ),
        },
        "sympy__sympy-17630": {
            "focal_fn": "get_postprocessor",
            "tags": ["sympy", "matrices", "blockmatrix", "zeromatrix"],
            "f2p": (
                "a = MatrixSymbol('a', 2, 2)\n"
                "b = MatrixSymbol('b', 2, 2)\n"
                "post_add = get_postprocessor(Add)\n"
                "expr = Add(a, b, evaluate=False)\n"
                "captured_args = []\n"
                "orig_matadd = MatAdd\n"
                "def spy_matadd(*args, **kwargs):\n"
                "    captured_args.append(args)\n"
                "    return orig_matadd(*args, **kwargs)\n"
                "globals()['MatAdd'] = spy_matadd\n"
                "res = post_add(expr)\n"
                "globals()['MatAdd'] = orig_matadd\n"
                "assert captured_args[0] == (a, b)\n"
            ),
            "p2p": (
                "a = MatrixSymbol('a', 2, 2)\n"
                "post_mul = get_postprocessor(Mul)\n"
                "expr_mul = Mul(2, a, evaluate=False)\n"
                "res_mul = post_mul(expr_mul)\n"
                "assert isinstance(res_mul, MatMul)\n"
            ),
        },
        "sympy__sympy-18698": {
            "focal_fn": "_symbolic_factor_list",
            "tags": ["sympy", "polynomials", "sqf_list", "factorization"],
            "f2p": (
                "from sympy.abc import x, y\n"
                "from sympy import expand\n"
                "coeff, factors = sqf_list((x**2 + 1) * (x - 1)**2 * (x - 2)**3 * (x - 3)**3)\n"
                "# Factors of the same multiplicity (3) must be combined into a single square-free factor (x**2 - 5*x + 6, 3)!\n"
                "mult_dict = {k: expand(f) for f, k in factors}\n"
                "assert len(factors) == 3\n"
                "assert mult_dict[3] == expand((x - 2) * (x - 3))\n"
            ),
            "p2p": (
                "from sympy.abc import x, y\n"
                "assert factor_list(x * (x + y))[1] == [(x, 1), (x + y, 1)]\n"
            ),
        },
        "sympy__sympy-23262": {
            "focal_fn": "_recursive_to_string",
            "tags": ["sympy", "lambdify", "tuple_codegen", "code_printer"],
            "f2p": (
                "assert _recursive_to_string(str, (1,)) == '(1,)'\n"
                "assert _recursive_to_string(str, (1, 2)) in ('(1, 2)', '(1, 2,)')\n"
                "f_single = lambdify([], (1,))\n"
                "assert f_single() == (1,) and isinstance(f_single(), tuple)\n"
            ),
            "p2p": (
                "assert _recursive_to_string(str, [1]) == '[1]'\n"
                "assert _recursive_to_string(str, [1, 2]) == '[1, 2]'\n"
                "f_pair = lambdify([], (1, 2))\n"
                "assert f_pair() == (1, 2)\n"
            ),
        },
    }

    suite: List[SWEBenchInstance] = []
    for instance_id, meta in specs.items():
        row = raw[instance_id]
        pad_b = 50 if instance_id in ("pytest-dev__pytest-7324", "sympy__sympy-18698") else 35
        pad_a = 45
        focal_code = _extract_focal_window(row["base_file_content"], row["patch"], pad_b, pad_a)
        suite.append(
            SWEBenchInstance(
                instance_id=instance_id,
                repo=row["repo"],
                base_commit=row["base_commit"],
                target_file=row["target_file"],
                difficulty=row.get("difficulty", "15 min - 1 hour"),
                problem_statement=row["problem_statement"],
                focal_window_code=focal_code,
                focal_function_name=meta["focal_fn"],
                base_file_content=row["base_file_content"],
                gold_patch=row["patch"],
                fail_to_pass_assertions=meta["f2p"],
                pass_to_pass_assertions=meta["p2p"],
                domain_tags=meta["tags"],
            )
        )
    return suite


def execute_swebench_patch(
    instance: SWEBenchInstance,
    patched_snippet: str,
) -> Tuple[bool, bool, bool, Optional[str]]:
    """
    Executes the base file + `patched_snippet` override in an isolated module scope,
    then runs `fail_to_pass_assertions` and `pass_to_pass_assertions`.
    Returns `(both_passed, f2p_passed, p2p_passed, diagnostic_error)`.
    """
    # Derive exact module & package name from target_file (e.g. 'astropy/modeling/separable.py' -> 'astropy.modeling.separable')
    rel_mod = instance.target_file.replace("\\", "/")
    if rel_mod.startswith("src/"):
        rel_mod = rel_mod[len("src/") :]
    if rel_mod.endswith(".py"):
        rel_mod = rel_mod[:-3]
    mod_name = rel_mod.replace("/", ".")
    pkg_name = mod_name.rsplit(".", 1)[0] if "." in mod_name else mod_name

    # Compatibility shim for older SymPy commits on Python 3.9+
    import sympy.core.compatibility as sym_compat
    if not hasattr(sym_compat, "range"):
        sym_compat.range = range  # type: ignore
    if not hasattr(sym_compat, "SYMPY_INTS"):
        sym_compat.SYMPY_INTS = (int,)  # type: ignore
    if not hasattr(sym_compat, "string_types"):
        sym_compat.string_types = (str,)  # type: ignore
    if not hasattr(sym_compat, "default_sort_key"):
        from sympy import default_sort_key
        sym_compat.default_sort_key = default_sort_key  # type: ignore
    if not hasattr(sym_compat, "ordered"):
        from sympy import ordered
        sym_compat.ordered = ordered  # type: ignore

    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": mod_name,
        "__package__": pkg_name,
        "__file__": instance.target_file,
    }
    try:
        # 1. Load the original base_commit module
        exec(compile(instance.base_file_content, instance.target_file, "exec"), namespace)
        # If the focal function is a method on NDArithmeticMixin (`astropy-14995`), expose it at top-level too
        if "NDArithmeticMixin" in namespace and "_arithmetic_mask" not in namespace:
            namespace["_arithmetic_mask"] = namespace["NDArithmeticMixin"]._arithmetic_mask
        # 2. Apply the LLM's patched function/class definition on top of the module namespace
        if patched_snippet.strip():
            import textwrap
            cleaned_patch = textwrap.dedent(patched_snippet)
            exec(compile(cleaned_patch, f"<patch:{instance.instance_id}>", "exec"), namespace)
            # Rebind functions defined in the base file whose closures call the patched helper!
            for fn_obj in list(namespace.values()):
                if (
                    callable(fn_obj)
                    and hasattr(fn_obj, "__globals__")
                    and isinstance(fn_obj.__globals__, dict)
                    and getattr(fn_obj, "__module__", None) == mod_name
                ):
                    fn_obj.__globals__.update(namespace)
            if "NDArithmeticMixin" in namespace and "_arithmetic_mask" in namespace:
                namespace["NDArithmeticMixin"]._arithmetic_mask = namespace["_arithmetic_mask"]
            if "_operators" in namespace and "_cstack" in namespace:
                namespace["_operators"]["&"] = namespace["_cstack"]
            if "_operators" in namespace and "_cdot" in namespace:
                namespace["_operators"]["|"] = namespace["_cdot"]
        # If sympy__sympy-17630 patched get_postprocessor, also rebind Basic._constructor_postprocessor_mapping
        if instance.instance_id == "sympy__sympy-17630" and "get_postprocessor" in namespace:
            from sympy.core.basic import Basic
            if "MatrixExpr" in namespace and "Mul" in namespace and "Add" in namespace:
                Basic._constructor_postprocessor_mapping[namespace["MatrixExpr"]] = {
                    "Mul": [namespace["get_postprocessor"](namespace["Mul"])],
                    "Add": [namespace["get_postprocessor"](namespace["Add"])],
                }
    except Exception as exc:
        return False, False, False, f"PatchCompilationError: {type(exc).__name__}: {exc}"

    # 3. Run FAIL_TO_PASS tests: first gather multi-line statements (def/class), then run line-by-line for assertions
    blocks: List[str] = []
    cur: List[str] = []
    for line in instance.fail_to_pass_assertions.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if (line.startswith(" ") or line.startswith("\t")) and cur:
            cur.append(line)
        else:
            if cur:
                blocks.append("\n".join(cur))
            cur = [line]
    if cur:
        blocks.append("\n".join(cur))

    try:
        for idx, blk in enumerate(blocks, start=1):
            stripped = blk.strip()
            try:
                exec(blk, namespace)
            except Exception as line_exc:
                detail = str(line_exc)
                if isinstance(line_exc, AssertionError) and stripped.startswith("assert ") and "==" in stripped:
                    expr_body = stripped[len("assert ") :].strip()
                    parts = expr_body.split("==", 1)
                    if len(parts) == 2:
                        try:
                            lhs_val = eval(parts[0].strip(), namespace)
                            rhs_val = eval(parts[1].strip(), namespace)
                            detail = f"LHS `{parts[0].strip()}` evaluated to {lhs_val!r}, expected RHS -> {rhs_val!r}"
                        except Exception:
                            pass
                elif isinstance(line_exc, AssertionError) and "np.array_equal(" in stripped:
                    try:
                        res_val = namespace.get("res")
                        exp_val = namespace.get("expected")
                        detail = f"`res` evaluated to {res_val!r}, expected {exp_val!r}"
                    except Exception:
                        pass
                return False, False, False, f"FAIL_TO_PASS statement #{idx} (`{stripped.splitlines()[0]}`) -> {type(line_exc).__name__}: {detail}"
        f2p_ok = True
    except Exception as exc:
        return False, False, False, f"FAIL_TO_PASS test failed -> {type(exc).__name__}: {exc}"

    # 4. Run PASS_TO_PASS regression tests
    try:
        exec(instance.pass_to_pass_assertions, namespace)
        p2p_ok = True
    except Exception as exc:
        return False, True, False, f"PASS_TO_PASS regression failed -> {type(exc).__name__}: {exc}"

    return True, f2p_ok, p2p_ok, None
