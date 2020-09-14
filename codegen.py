from abc import ABC, abstractmethod
from nodes import Node

class CodeGenerator(ABC):
    @abstractmethod
    def generate_code(self, node: Node, indent_level: int = 0, generate_pass: bool = False) -> str:
        pass

