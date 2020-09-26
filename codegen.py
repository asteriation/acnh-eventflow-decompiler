from abc import ABC, abstractmethod
from nodes import Node
from predicates import Predicate

from typing import Callable, Union

NodeCodeGenerator = Callable[[Node, int, bool], str]
PredicateCodeGenerator = Callable[[Predicate], str]

class CodeGenerator(ABC):
    @abstractmethod
    def generate_actor_annotation(self, actor_name: str, secondary_name: str) -> str:
        pass

    @abstractmethod
    def generate_code(self, node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
        pass

