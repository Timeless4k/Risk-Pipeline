from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Unified model interface for fair evaluation.

    Implementations should not change data influence.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_model(self, input_spec: Any) -> None:
        ...

    @abstractmethod
    def fit(self, X: Any, y: Any, config: Any) -> None:
        ...

    @abstractmethod
    def predict(self, X: Any) -> Any:
        ...
