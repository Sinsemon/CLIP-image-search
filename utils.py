from time import time_ns
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self

class catch_time(AbstractContextManager):
    start:int

    def __init__(self, should_print:bool = False, name:str = "Time") -> None:
        self.start = time_ns()
        self._should_print = should_print
        self._name = name

    def __enter__(self) -> Self:
        self.start = time_ns()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._should_print:
            print(f"{self._name}: {self.duration}")
        return None

    @property
    def duration(self) -> float:
        return (time_ns() - self.start) / 1e9

    

