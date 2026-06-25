"""A restricted boolean/arithmetic expression evaluator.

Used as Policy's deterministic fallback when no LLM provider is active.
Policy statements are otherwise judged in natural language by an LLM, but
that requires a configured ModelBinding; this evaluator lets a policy with
a `fallback_expression` (e.g. ``"purchase_amount <= approval_limit"``) keep
working offline, without ever passing a policy string to ``eval``/``exec``
(which would let it execute arbitrary code). It walks the parsed AST
instead and only permits literals, names, comparisons, boolean logic and
basic arithmetic.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_COMPARES: dict[type, Any] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_UNARYOPS: dict[type, Any] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains a construct outside the allowed grammar."""


class MissingVariableError(UnsafeExpressionError):
    """Raised when an expression references a name absent from the given context."""


def safe_eval(expression: str, variables: dict[str, Any]) -> Any:
    """Evaluate `expression` using only literals, names, comparisons,
    boolean operators and arithmetic -- no calls, attributes or subscripts."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        raise UnsafeExpressionError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise MissingVariableError(f"Unknown variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = result and _eval_node(value, variables)
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = result or _eval_node(value, variables)
            return result
        raise UnsafeExpressionError("Unsupported boolean operator")
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError("Unsupported unary operator")
        return op(_eval_node(node.operand, variables))
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError("Unsupported binary operator")
        return op(_eval_node(node.left, variables), _eval_node(node.right, variables))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        result = True
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARES.get(type(op_node))
            if op is None:
                raise UnsafeExpressionError("Unsupported comparison operator")
            right = _eval_node(comparator, variables)
            result = result and op(left, right)
            left = right
        return result
    raise UnsafeExpressionError(f"Unsupported expression element: {type(node).__name__}")
