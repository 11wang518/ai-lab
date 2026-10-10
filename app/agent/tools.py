"""智能体可调用的工具集。

只使用标准库实现，避免为了一个小功能引入依赖。
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
import re
from typing import Any, Callable

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字运算")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的运算符")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("不支持的一元运算符")
        return op(_eval_node(node.operand))
    raise ValueError("表达式过于复杂，已拒绝执行")


def calculator(expression: str) -> dict:
    """安全计算四则运算表达式，不使用 eval。"""
    cleaned = _clean_expression(expression)
    if not cleaned:
        raise ValueError("表达式为空")
    if len(cleaned) > 200:
        raise ValueError("表达式过长")
    value = _eval_node(_parse(cleaned))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"expression": expression, "normalized": cleaned, "value": value}


def _clean_expression(expression: str) -> str:
    return expression.strip().replace("×", "*").replace("÷", "/").replace("^", "**")


def _parse(cleaned: str) -> ast.AST:
    """解析表达式；从文本里截出来的片段可能带多余尾巴，逐字符回退。"""
    candidate = cleaned
    # 只做有限次回退：修复截断留下的尾巴，避免把残缺表达式"修"成一个错误答案。
    for _ in range(4):
        try:
            return ast.parse(candidate, mode="eval")
        except SyntaxError:
            trimmed = candidate.rstrip(" \t+-*/.^,;")
            if trimmed == candidate:
                trimmed = candidate[:-1]
            if not trimmed:
                break
            candidate = trimmed
    raise ValueError(f"无法解析的表达式：{cleaned}")


def current_time(timezone_label: str = "Asia/Shanghai") -> dict:
    """返回本机当前时间。"""
    now = _dt.datetime.now()
    return {
        "timezone": timezone_label,
        "iso": now.isoformat(timespec="seconds"),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": "一二三四五六日"[now.weekday()],
    }


def text_stats(text: str) -> dict:
    """统计一段文本的字符数、行数、中文字数。"""
    content = text or ""
    chinese = len(re.findall(r"[\u4e00-\u9fff]", content))
    return {
        "characters": len(content),
        "lines": content.count("\n") + (1 if content else 0),
        "chinese_characters": chinese,
        "words_ascii": len(re.findall(r"[A-Za-z0-9_]+", content)),
    }


TOOLS: dict[str, dict] = {
    "calculator": {
        "fn": calculator,
        "description": "计算一个数学表达式，例如 12*(3+4)。",
        "parameters": {"expression": "要计算的表达式"},
    },
    "current_time": {
        "fn": current_time,
        "description": "获取本机当前日期与时间。",
        "parameters": {"timezone_label": "时区标签，默认 Asia/Shanghai"},
    },
    "text_stats": {
        "fn": text_stats,
        "description": "统计文本的字符数、行数、中文字数。",
        "parameters": {"text": "要统计的文本"},
    },
}


def describe_tools() -> list[dict]:
    return [
        {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        }
        for name, spec in TOOLS.items()
    ]


def call_tool(name: str, arguments: dict) -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"没有名为 {name} 的工具")
    return spec["fn"](**arguments)
