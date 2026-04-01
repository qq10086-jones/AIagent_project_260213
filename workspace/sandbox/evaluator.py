#!/usr/bin/env python3
"""AST evaluator for calculator."""

import ast
import math
import operator


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"Unsupported symbol: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        return float(_BIN_OPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        value = evaluate_node(node.operand)
        return float(_UNARY_OPS[op_type](value))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [evaluate_node(arg) for arg in node.args]
        return float(_ALLOWED_FUNCS[func_name](*args))

    raise ValueError(f"Unsupported expression: {type(node).__name__}")
