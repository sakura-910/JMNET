from .resnet import resnet18
#from .resnet18 import eca_resnet18
from .builder import build_backbone
from .eca_module import eca_layer
from .eca_resnet18 import eca_resnet18, eca_resnet34
#__all__ = ['eca_resnet18', 'resnet50', 'resnet101']
#__all__ = ['resnet18']
__all__ = ['eca_resnet18', 'eca_resnet34', 'resnet18']

