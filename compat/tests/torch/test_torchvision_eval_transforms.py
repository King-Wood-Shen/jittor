"""PIL resize geometry and diagnostic repr contracts used by timm eval."""
import pytest
import torch
from PIL import Image
from torchvision.transforms import Resize, Normalize

@pytest.mark.parametrize('wh,size,max_size,expected', [
    ((1682,694),235,None,(569,235)),
    ((694,1682),235,None,(235,569)),
    ((7,3),2,None,(4,2)),
    ((3,7),2,None,(2,4)),
    ((7,3),[2],None,(4,2)),
    ((7,3),(2,),None,(4,2)),
    ((7,3),(2,5),None,(5,2)),
    ((13,5),3,6,(6,2)),
    ((5,13),[3],6,(2,6)),
    ((13,5),3,9,(7,3)),
    ((13,5),5,None,(13,5)),
])
def test_resize_pil_geometry(wh,size,max_size,expected):
    image=Image.new('RGB',wh,(17,83,201))
    result=Resize(size,max_size=max_size)(image)
    assert result.size==expected
    assert result.mode=='RGB' and result.getpixel((0,0))==(17,83,201)

@pytest.mark.parametrize('size,max_size,error', [
    ([],None,ValueError),([2,3,4],None,ValueError),(2.5,None,TypeError),
    (2,2,ValueError),(2,1,ValueError),([2,3],6,ValueError),
])
def test_resize_rejects_invalid_size_contract(size,max_size,error):
    with pytest.raises(error):
        Resize(size,max_size=max_size)(Image.new('RGB',(7,3)))

@pytest.mark.parametrize('container',[list,tuple])
def test_normalize_sequence_repr(container):
    mean=container([0.485,0.456,0.406]);std=container([0.229,0.224,0.225])
    assert repr(Normalize(mean,std))=='Normalize(mean='+repr(mean)+', std='+repr(std)+')'

def test_normalize_tensor_repr_preserves_values():
    mean=torch.tensor([0.485,0.456,0.406]);std=torch.tensor([0.229,0.224,0.225])
    before=(mean.tolist(),std.tolist())
    text=repr(Normalize(mean,std))
    assert text.startswith('Normalize(mean=') and ', std=' in text
    assert '0.485' in text and '0.229' in text
    assert (mean.tolist(),std.tolist())==before
