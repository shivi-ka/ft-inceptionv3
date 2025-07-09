from torchvision import models
import torch.nn as nn


def create_model():
    # --- Load pre-trained Inception-V3 ---
    model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')

    # --- Freeze all base layers ---
    for param in model.parameters():
        param.requires_grad = False

    # --- Replace the final classifier (and auxiliary classifier) ---
    # The original InceptionV3 has an auxiliary output, we need to handle both
    # Main Classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) # Output is 1 for binary classification
    
    # Auxiliary Classifier
    num_ftrs_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs_aux, 1)

    return model