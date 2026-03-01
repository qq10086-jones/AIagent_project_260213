#!/usr/bin/env python3
"""Expression parser for calculator."""

import ast


def parse_expression(expression: str) -> ast.Expression:
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("Expression is empty")
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e.msg}") from e
    return parsed
