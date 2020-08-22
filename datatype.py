from __future__ import annotations

from typing import Any

class Argument(str):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        str.__init__(*args, **kwargs)

class Type:
    def __init__(self, type_: str) -> None:
        self.type = type_

    def format(self, value: Any) -> str:
        if isinstance(value, Argument):
            return str(value)

        if self.type.startswith('int'):
            assert isinstance(value, int)
            if self.type != 'int':
                n = int(self.type[3:])
                assert 0 <= value < n
            return repr(int(value))
        elif self.type.startswith('enum'):
            assert not isinstance(value, bool) and isinstance(value, int)
            vals = self.type[5:-1].split(',')
            assert 0 <= value < len(vals)
            return vals[value]
        elif self.type == 'float':
            assert not isinstance(value, bool) and isinstance(value, (int, float))
            return repr(float(value))
        elif self.type == 'str':
            assert isinstance(value, str)
            return repr(value)
        elif self.type == 'bool':
            assert isinstance(value, bool) or (isinstance(value, int) and 0 <= value <= 1)
            return 'true' if value else 'false'
        elif self.type == 'any':
            return repr(value)
        else:
            raise ValueError(f'bad type: {self.type}')

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

