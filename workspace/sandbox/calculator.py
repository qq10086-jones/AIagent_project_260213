#!/usr/bin/env python3
"""
Calculator test scaffold for Coder mode.
Use this file as the target for /coder patch tasks.
"""

from evaluator import evaluate_node
from parser import parse_expression


def calculate(expression: str) -> float:
    parsed = parse_expression(expression)
    return evaluate_node(parsed)


def main() -> None:
    expr = input("Enter expression (e.g. sin(pi/2) + 12/3): ").strip()
    try:
        result = calculate(expr)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
