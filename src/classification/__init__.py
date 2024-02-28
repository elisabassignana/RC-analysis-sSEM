from .classifiers import *
from .losses import *


def load_classifier(double_entities_markers=False):

    if double_entities_markers:
        return MultiLayerPerceptronClassifierDouble, LabelLoss
    else:
        return MultiLayerPerceptronClassifier, LabelLoss

