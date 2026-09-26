"""Name-based plugin registries.

Builtins register themselves at import. A hot-plugged module does the same
by importing ``registry`` and using the ``@strategies.register("name")``
decorator. Re-registering a name replaces the previous class so a plugin
file can be reloaded while iterating.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            key = name or getattr(cls, "name", None) or cls.__name__
            self._items[key] = cls
            cls.name = key  # type: ignore[attr-defined]
            return cls

        return deco

    def create(self, name: str, **kwargs: object) -> T:
        try:
            cls = self._items[name]
        except KeyError as exc:
            known = ", ".join(self.names()) or "(empty)"
            raise KeyError(
                f"unknown {self.kind} {name!r}. available: {known}"
            ) from exc
        return cls(**kwargs)  # type: ignore[call-arg]

    def names(self) -> list[str]:
        return sorted(self._items)

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(self.names()) or "(empty)"
            raise KeyError(
                f"unknown {self.kind} {name!r}. available: {known}"
            ) from exc


metrics: Registry[object] = Registry("metric")
processes: Registry[object] = Registry("process")
policies: Registry[object] = Registry("policy")
strategies: Registry[object] = Registry("strategy")
effects: Registry[object] = Registry("effect")
