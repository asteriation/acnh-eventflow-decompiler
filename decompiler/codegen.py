from abc import ABC, abstractmethod

from typing import Callable, Union

from .nodes import Node
from .predicates import Predicate

NodeCodeGenerator = Callable[[Node, int, bool], str]
PredicateCodeGenerator = Callable[[Predicate], str]

class CodeGenerator(ABC):
    @abstractmethod
    def generate_code(self, node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
        pass

