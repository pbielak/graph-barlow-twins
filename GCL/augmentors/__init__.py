from .augmentor import Graph, Augmentor, Compose, RandomChoice
from .identity import Identity
from .ppr_diffusion import PPRDiffusion
from .edge_removing import EdgeRemoving
from .feature_masking import FeatureMasking

__all__ = [
    'Graph',
    'Augmentor',
    'Compose',
    'RandomChoice',
    'EdgeRemoving',
    'FeatureMasking',
    'Identity',
    'PPRDiffusion',
]

classes = __all__
