import torch

def corn_label_from_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORN.
    """
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def coral_label_from_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORAL.
    """
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def crossentr_label_from_logits(logits):
    _, predicted_labels = torch.max(logits, 1)
    return predicted_labels


def niu_label_from_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORAL.
    """
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels