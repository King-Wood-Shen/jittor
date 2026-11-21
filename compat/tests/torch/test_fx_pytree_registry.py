import unittest

import torch


class _LeafSpec:
    def is_leaf(self):
        return True


class _Box:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class _BoxSpec:
    type = _Box

    def is_leaf(self):
        return False

    def children(self):
        return [_LeafSpec(), _LeafSpec()]


class TestFxPytreeRegistry(unittest.TestCase):
    def test_registered_flattener_returns_real_leaves(self):
        from torch.fx import _pytree

        _pytree.register_pytree_flatten_spec(
            _Box, lambda value, spec: [value.left, value.right])
        self.assertEqual(
            _pytree.tree_flatten_spec(_Box(3, 7), _BoxSpec()), [3, 7])

    def test_wrong_child_count_fails_closed(self):
        from torch.fx import _pytree

        _pytree.register_pytree_flatten_spec(
            _Box, lambda value, spec: [value.left])
        with self.assertRaises(ValueError):
            _pytree.tree_flatten_spec(_Box(3, 7), _BoxSpec())


if __name__ == '__main__':
    unittest.main()
