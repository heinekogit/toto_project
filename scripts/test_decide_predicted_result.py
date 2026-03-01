#!/usr/bin/env python3
import os
import sys
import ast

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd


MODULE_PATH = os.path.join(ROOT_DIR, "scripts", "11_prediction_01.py")


def _load_decide_function(path: str):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    mod = ast.parse(src, filename=path)
    fn_node = None
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "decide_predicted_result":
            fn_node = node
            break
    if fn_node is None:
        raise RuntimeError("decide_predicted_result not found")

    isolated = ast.Module(body=[fn_node], type_ignores=[])
    code = compile(isolated, filename=path, mode="exec")
    ns = {"pd": pd}
    exec(code, ns)
    return ns["decide_predicted_result"]


decide_predicted_result = _load_decide_function(MODULE_PATH)


def test_decide_predicted_result_basic():
    assert decide_predicted_result(0.60, 0.33, 0.07) == "H"
    assert decide_predicted_result(0.21, 0.33, 0.46) == "A"
    assert decide_predicted_result(0.30, 0.34, 0.36) == "A"
    assert decide_predicted_result(0.33, 0.34, 0.33) == "D"


if __name__ == "__main__":
    test_decide_predicted_result_basic()
    print("ok")
