from collections import OrderedDict
import pytest
import torch

@pytest.mark.parametrize('method',['mapping','pairs','add_module','setitem','update'])
def test_registration_paths_share_order_and_none(method):
    a,b=torch.nn.Linear(2,2),torch.nn.Linear(2,2)
    pairs=[('a',a),('b',b)]
    if method=='mapping': m=torch.nn.ModuleDict(OrderedDict(pairs))
    elif method=='pairs': m=torch.nn.ModuleDict(pairs)
    else:
        m=torch.nn.ModuleDict()
        if method=='add_module':
            for key,value in pairs: assert m.add_module(key,value) is None
        elif method=='setitem':
            for key,value in pairs: m[key]=value
        else: m.update(pairs)
    assert list(m)==list(m.keys())==[k for k,_ in m.items()]==['a','b']
    assert list(m._modules)==['a','b']
    assert [k for k,_ in m.named_children()]==['a','b']
    assert set(m.state_dict())=={'a.weight','a.bias','b.weight','b.bias'}
    replacement=torch.nn.Linear(2,2)
    m.add_module('a',replacement)
    assert m['a'] is replacement and list(m)==['a','b']
    m.add_module('empty',None)
    assert m['empty'] is None and list(m)==['a','b','empty'] and len(m)==3
    assert list(m._modules)==['a','b','empty']
    assert [k for k,_ in m.named_children()]==['a','b']
    assert m.pop('b') is b
    assert list(m)==['a','empty'] and list(m._modules)==['a','empty']
    m.add_module('b',b)
    assert list(m)==['a','empty','b'] and list(m._modules)==['a','empty','b']
    with pytest.raises(KeyError): m['absent']
    with pytest.raises(KeyError): del m['absent']

@pytest.mark.parametrize('method',['add_module','setitem','update'])
@pytest.mark.parametrize('key,value_kind,error',[
    ('','module',KeyError),('a.b','module',KeyError),(3,'module',TypeError),
    ('items','module',KeyError),('bad','int',TypeError)])
def test_invalid_registration_is_atomic(method,key,value_kind,error):
    m=torch.nn.ModuleDict({'valid':torch.nn.Identity()})
    value=torch.nn.Identity() if value_kind=='module' else 3
    with pytest.raises(error):
        if method=='add_module': m.add_module(key,value)
        elif method=='setitem': m[key]=value
        else: m.update([(key,value)])
    assert list(m)==['valid'] and [k for k,_ in m.named_children()]==['valid']

def test_subclass_forward_iterates_added_children():
    class Add(torch.nn.Module):
        def forward(self,x): return x+1
    class Block(torch.nn.ModuleDict):
        def __init__(self):
            super().__init__()
            self.add_module('first',Add());self.add_module('second',Add())
        def forward(self,x):
            for _,child in self.items(): x=child(x)
            return x
    block=Block()
    assert block(torch.tensor([1.])).item()==3
    assert block.forward(torch.tensor([1.])).item()==3

def test_nested_module_dict_keeps_state_and_iteration_consistent():
    inner=torch.nn.ModuleDict()
    inner.add_module('linear',torch.nn.Linear(2,3))
    outer=torch.nn.ModuleDict()
    outer.add_module('block',inner)
    assert list(outer)==['block'] and list(outer['block'])==['linear']
    assert set(outer.state_dict())=={'block.linear.weight','block.linear.bias'}
