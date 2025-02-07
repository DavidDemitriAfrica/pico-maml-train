import torch.nn as nn
import learn2learn as l2l


def clone_classifier_params(model: nn.Module) -> nn.Module:
    """
    Clone the classifier module using learn2learn.
    (Note: With learn2learn you can work directly with the cloned module.)
    """
    return l2l.clone_module(model.classifier)
