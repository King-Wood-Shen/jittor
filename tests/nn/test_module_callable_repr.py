"""Module representation supports constructor descriptors and native callables."""
import types
import unittest
import jittor as jt


class _Initializer:
    def __init__(self, fn):
        self.__wrapped__ = fn

    def __get__(self, instance, owner=None):
        return self if instance is None else types.MethodType(self, instance)

    def __call__(self, module, *args, **kwargs):
        self.__wrapped__(module, *args, **kwargs)


class _Plain(jt.Module):
    def __init__(self, width):
        self.width = width


class _Wrapped(jt.Module):
    @_Initializer
    def __init__(self, width):
        self.width = width


class _Builtin(jt.Module):
    __init__ = object.__init__


class TestModuleCallableRepr(unittest.TestCase):
    def test_wrapped_initializer_matches_plain(self):
        self.assertEqual(_Wrapped(7).extra_repr(), _Plain(7).extra_repr())
        self.assertIn("7", str(_Wrapped(7)))

    def test_builtin_initializer_has_empty_extra_repr(self):
        self.assertEqual(_Builtin().extra_repr(), "")
        self.assertIn("7", _Wrapped(7).extra_repr())
