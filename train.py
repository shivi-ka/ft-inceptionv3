from src import config
from src.dataset import get_dataloaders
from src.model import create_model
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from src.trainer import train_model
from src.evaluate import evaluate_model


def main():
    # Get Device
    device = config.DEVICE

    # Data Loading
    dataloader, train_labels = get_dataloaders(data_path=config.DS_PATH, batch_size=config.BATCH_SIZE)

    # Define Model
    model = create_model()

    # Define Loss Function
    counts = np.bincount(train_labels.flatten()) # [ 388 3494]
    pos_weight = torch.tensor(counts[0] / counts[1], dtype=torch.float).to(device)
    print(f"Class imbalance weight (for pneumonia class): {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---Step 1 of 2 for traning the model (training classifier head)---
    
    print("\n--- STAGE 1: Training the Classifier Head ---")
    params_to_update_head = [] # List to hold parameters that will be updated
    # Only train the new layers (fc and AuxLogits.fc)
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update_head.append(param)
            print(f"\tTraining: {name}")
    # Define Optimizer for classifier head
    optimizer_head = optim.Adam(params_to_update_head, lr=config.LEARNING_RATE_HEAD)

    # Train the model with the new head
    model = train_model(device=device,
                        dataloaders=dataloader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer_head,
                        num_epochs=config.NUM_EPOCHS_HEAD,
                        patience=config.PATIENCE)
    
    # ---Step 2 of 2 for training the model (finetuning)---
    print("\n--- STAGE 2: Fine-Tuning the Full Model ---")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Define Optimizer for fine-tuning (with very small learning rate)
    optimizer_finetune = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_FINETUNE)
    # Train the full model with fine-tuning
    model = train_model(device=device,
                        dataloaders=dataloader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer_finetune,
                        num_epochs=config.NUM_EPOCHS_FINETUNE,
                        patience=config.PATIENCE)
    
    # Save the best model
    print(f"\n--- Saving Model to {config.MODEL_SAVE_PATH} ---")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print("Model saved successfully.")

    # Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    evaluate_model(device=device,
                   dataloaders=dataloader,
                   model=model,
                   threshold=config.THRESHOLD)

if __name__ == "__main__":
    main()