from __future__ import annotations

from typing import Any

class Argument(str):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        str.__init__(*args, **kwargs)

class Type:
    def __init__(self, type_: str) -> None:
        self.type = type_

    def num_values(self) -> int:
        if self.type.startswith('int') and self.type != 'int':
            return int(self.type[3:])
        elif self.type == 'bool':
            return 2
        elif self.type.startswith('enum'):
            return len(self.type.split(','))
        else:
            return 999999999

    def __str__(self) -> str:
        return self.type

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Type):
            return False
        return self.type == other.type

    def __hash__(self) -> int:
        return hash(self.type)

AnyType = Type('any')
BoolType = Type('bool')
FloatType = Type('float')
IntType = Type('int')
StrType = Type('str')

