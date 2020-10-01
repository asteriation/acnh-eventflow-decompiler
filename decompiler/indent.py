from __future__ import annotations

INDENT_NUM_SPACES = 4

def indent(level: int) -> str:
    return ' ' * (INDENT_NUM_SPACES * level)

