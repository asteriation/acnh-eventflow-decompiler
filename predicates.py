from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from datatype import BoolType
from actors import Query

class Predicate(ABC):
    @abstractmethod
    def generate_code(self) -> str:
        pass

    def hint(self) -> List[str]:
        return []

    def __invert__(self) -> Predicate:
        return NotPredicate(self)

    def __and__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)
        if isinstance(other, AndPredicate):
            return AndPredicate([self]) & other
        return AndPredicate([self, other])

    def __or__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)
        if isinstance(other, OrPredicate):
            return OrPredicate([self]) | other
        return OrPredicate([self, other])

class ConstPredicate(Predicate):
    def __init__(self, value: bool) -> None:
        self.value = value

    def generate_code(self) -> str:
        return BoolType.format(self.value)

class QueryPredicate(Predicate):
    def __init__(self, query: Query, params: Dict[str, Any], values: List[Any]) -> None:
        assert len(values) > 0

        self.query = query
        self.params = params
        self.values = values
        self.negated = False

        if self.query.rv == BoolType and self.values == [False]:
            self.negated = True

        if query.inverted:
            self.negated = not self.negated

    def generate_code(self) -> str:
        if self.query.rv == BoolType:
            if len(self.values) == 1:
                return self.query.format(self.params, self.negated)
            else:
                return 'False' if self.negated else 'True'
        else:
            if len(self.values) == 1:
                op = '!=' if self.negated else '=='
                try:
                    return f'{self.query.format(self.params, False)} {op} {self.query.rv.format(self.values[0])}'
                except:
                    print(self.query, self.values)
                    raise
            else:
                op = 'not in' if self.negated else 'in'
                try:
                    vals_s = [self.query.rv.format(v) for v in self.values]
                except:
                    print(self.query, self.values)
                    raise
                return f'{self.query.format(self.params, False)} {op} ({", ".join(vals_s)})'

    def hint(self) -> List[str]:
        return self.query.hint(self.params)

    def __invert__(self) -> Predicate:
        qp = QueryPredicate(self.query, self.params, self.values)
        qp.negated = not self.negated
        return qp

class NotPredicate(Predicate):
    def __init__(self, inner: Predicate) -> None:
        self.inner = inner

    def generate_code(self) -> str:
        return f'not ({self.inner.generate_code()})'

    def __invert__(self) -> Predicate:
        return self.inner

class AndPredicate(Predicate):
    def __init__(self, inners: List[Predicate]) -> None:
        self.inners = inners

    def generate_code(self) -> str:
        return ' and '.join([f'({inner.generate_code()})' for inner in self.inners])

    def __invert__(self) -> Predicate:
        # if the majority of inner predicates are negated, convert to or
        num_not_predicates = sum(
            isinstance(p, NotPredicate) or (isinstance(p, QueryPredicate) and p.negated)
            for p in self.inners
        )
        if num_not_predicates * 2 >= len(self.inners):
            return OrPredicate([~p for p in self.inners])
        return Predicate.__invert__(self)

    def __and__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)

        if isinstance(other, AndPredicate):
            return AndPredicate(self.inners + other.inners)
        else:
            return AndPredicate(self.inners + [other])

class OrPredicate(Predicate):
    def __init__(self, inners: List[Predicate]) -> None:
        self.inners = inners

    def generate_code(self) -> str:
        return ' or '.join([f'({inner.generate_code()})' for inner in self.inners])

    def __invert__(self) -> Predicate:
        # if the majority of inner predicates are negated, convert to or
        num_not_predicates = sum(1 for p in self.inners if isinstance(p, NotPredicate) or (isinstance(p, QueryPredicate) and p.negated))
        if num_not_predicates * 2 >= len(self.inners):
            return AndPredicate([~p for p in self.inners])
        return Predicate.__invert__(self)

    def __or__(self, other: Predicate) -> Predicate:
        assert isinstance(other, Predicate)

        if isinstance(other, OrPredicate):
            return OrPredicate(self.inners + other.inners)
        else:
            return OrPredicate(self.inners + [other])

