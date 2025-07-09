import torch

# Set device to GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE_HEAD = 1e-3  # Learning rate for the new classifier head
LEARNING_RATE_FINETUNE = 1e-5 # Lower learning rate for fine-tuning the whole model
NUM_EPOCHS_HEAD = 5        # Epochs for training the head
NUM_EPOCHS_FINETUNE = 10     # Epochs for fine-tuning
PATIENCE = 3               # Early stopping patience
AUX_LOSS_WEIGHT = 0.4        # Weight for auxiliary loss in InceptionV3 paper
PREDICITION_THRESHOLD = 0.5  # Threshold for binary classification

# File Paths 
DS_PATH = "data/raw/pneumoniamnist.npz"
MODEL_SAVE_PATH = "models/best_model.pth"

# Reproducibility
RANDOM_SEED = 42