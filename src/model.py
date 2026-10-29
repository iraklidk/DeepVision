import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(unfreeze_last_block=False):
    """Create ImageNet-pretrained ResNet-18 with a two-class output head."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Freeze all pretrained parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Optional experiment: also fine-tune ResNet's final feature block.
    if unfreeze_last_block:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    # A newly initialized two-class classifier is trainable by default.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model, weights


def set_training_modes(model, unfreeze_last_block=False):
    """Keep frozen BatchNorm running statistics from drifting."""
    model.train()
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()
    if not unfreeze_last_block:
        model.layer4.eval()
    model.fc.train()
