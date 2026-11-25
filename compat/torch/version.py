"""Torch frontend version values with string and release-sequence comparisons."""
from collections.abc import Iterable


class TorchVersion(str):
    __slots__ = ()

    def _compare(self, other, operation):
        from packaging.version import InvalidVersion, Version

        try:
            if not isinstance(other, (str, Version)):
                if not isinstance(other, Iterable):
                    return getattr(str, operation)(self, other)
                other = ".".join(map(str, other))
            version = other if isinstance(other, Version) else Version(other)
            return getattr(Version(self), operation)(version)
        except InvalidVersion:
            return getattr(str, operation)(self, other)


for operation in ("__lt__", "__le__", "__eq__", "__ge__", "__gt__"):
    setattr(TorchVersion, operation,
            lambda self, other, op=operation: self._compare(other, op))
