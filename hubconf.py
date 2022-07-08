from functools import partial

import torch

"""File for accessing the models via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    import torchvision.utils as vutils
    # Choose to use the device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load the model into the specified device.
    model = torch.hub.load("...", "...", pretrained=True, progress=True, verbose=False)
    model.eval()
    model = model.to(device)
"""

dependencies = ["torch"]


def resnet34_corn_afad(pretrained=False, progress=True):
    """
    ResNet34 ordinal regression model trained with CORN on AFAD
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from _train.helper import resnet34base

    NUM_CLASSES = 13
    model = resnet34base(
        num_classes=NUM_CLASSES, grayscale=False, resnet34_avg_poolsize=4
    )

    model.output_layer = torch.nn.Linear(512, NUM_CLASSES - 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.output_layer(x)
        return logits

    def add_method(obj, func):
        "Bind a function and store it in an object"
        setattr(obj, func.__name__, partial(func, obj))

    add_method(model, forward)

    if pretrained:
        checkpoint = (
            "https://github.com/rasbt/ord-torchhub/releases/"
            "download/1.0.0/resnet34_corn_afad.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, progress=True, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model


def resnet34_coral_afad(pretrained=False, progress=True):
    """
    ResNet34 ordinal regression model trained with CORAL on AFAD
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from _train.helper import resnet34base

    NUM_CLASSES = 13
    model = resnet34base(
        num_classes=NUM_CLASSES, grayscale=False, resnet34_avg_poolsize=4
    )

    model.output_layer = torch.nn.Linear(512, 1, bias=False)
    model.output_biases = torch.nn.Parameter(torch.zeros(NUM_CLASSES - 1).float())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.output_layer(x) + self.output_biases
        return logits

    def add_method(obj, func):
        "Bind a function and store it in an object"
        setattr(obj, func.__name__, partial(func, obj))

    add_method(model, forward)

    if pretrained:
        checkpoint = (
            "https://github.com/rasbt/ord-torchhub/"
            "releases/download/1.0.0/resnet34_coral_afad.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, progress=True, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model


def resnet34_niu_afad(pretrained=False, progress=True):
    """
    ResNet34 ordinal regression model trained with Niu et al.'s loss on AFAD
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from _train.helper import resnet34base

    NUM_CLASSES = 13
    model = resnet34base(
        num_classes=NUM_CLASSES, grayscale=False, resnet34_avg_poolsize=4
    )

    model.output_layer = torch.nn.Linear(512, NUM_CLASSES - 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.output_layer(x)
        return logits

    def add_method(obj, func):
        "Bind a function and store it in an object"
        setattr(obj, func.__name__, partial(func, obj))

    add_method(model, forward)

    if pretrained:
        checkpoint = (
            "https://github.com/rasbt/ord-torchhub/"
            "releases/download/1.0.0/resnet34_niu_afad.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, progress=True, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model


def resnet34_crossentr_afad(pretrained=False, progress=True):
    """
    ResNet34 ordinal regression model trained with regular Cross Entropy
      Loss on AFAD.
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    from _train.helper import resnet34base

    NUM_CLASSES = 13
    model = resnet34base(
        num_classes=NUM_CLASSES, grayscale=False, resnet34_avg_poolsize=4
    )

    model.output_layer = torch.nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.output_layer(x)
        return logits

    def add_method(obj, func):
        "Bind a function and store it in an object"
        setattr(obj, func.__name__, partial(func, obj))

    add_method(model, forward)

    if pretrained:
        checkpoint = (
            "https://github.com/rasbt/ord-torchhub/releases/"
            "download/1.0.0/resnet34_crossentr_afad.pt"
        )
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint, progress=True, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

    return model
