from typing import Any

class ConstBitStream:
    pos: int
    def __init__(self, bytes: bytes) -> None: ...
    def read(self, typ: str) -> Any: ...

