from src import config
from src.dataset import get_dataloaders, PneumoniaDataset
from src.model import create_model
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from src.trainer import train_model
from src.evaluate import evaluate_model, find_misclassified_indices, GradCAM
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms


torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def main():
    # Get Device
    device = config.DEVICE

    # Data Loading
    dataloaders, train_labels = get_dataloaders(data_path=config.DS_PATH, batch_size=config.BATCH_SIZE)

    # Define Model
    model = create_model()
    model = model.to(device)

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
                        dataloaders=dataloaders,
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
                        dataloaders=dataloaders,
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
                   dataloaders=dataloaders,
                   model=model,
                   threshold=config.PREDICITION_THRESHOLD)
    
    # Find misclassified indices
    # --- Main logic for analysis ---

    print("\n--- Finding a misclassified case for analysis ---")
    # Call the function we just defined
    misclassified = find_misclassified_indices(model, dataloaders['test'], device)

    if not misclassified['fn'] and not misclassified['fp']:
        print("No misclassifications found! Cannot perform analysis.")
    else:
        # Prioritize analyzing a False Negative, but fall back to a False Positive if none exist
        if misclassified['fn']:
            case_type = 'fn'
            case_idx = misclassified[case_type][0]
        else:
            case_type = 'fp'
            case_idx = misclassified[case_type][0]

        # --- Retrieve the image and its label ---
        data = np.load("data/raw/pneumoniamnist.npz")
        test_images, test_labels = data['test_images'], data['test_labels']
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.Grayscale(num_output_channels=3), # InceptionV3 needs 3 channels
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
        test_dataset = PneumoniaDataset(test_images, test_labels, transform=data_transforms['val']) # No augmentation for test

        image_tensor, label_tensor = test_dataset[case_idx]
        image_for_model = image_tensor.unsqueeze(0).to(device)
        original_image_np = test_dataset.images[case_idx] 

        model.eval()
        with torch.no_grad():
            pred_logit = model(image_for_model)
            pred_prob = torch.sigmoid(pred_logit).item()

        # --- Print details about the chosen case ---
        case_type_str = 'False Negative' if case_type == 'fn' else 'False Positive'
        true_label_str = 'Pneumonia' if label_tensor.item() == 1 else 'Normal'
        pred_label_str = 'Pneumonia' if pred_prob >= 0.5 else 'Normal'
        print(f"Analyzing Case Index: {case_idx}")
        print(f"Case Type: {case_type_str}")
        print(f"True Label: {true_label_str}")
        print(f"Model Predicted: {pred_label_str} (Probability: {pred_prob:.4f})")

        # --- Generate and Save Grad-CAM Visualization ---
        # InceptionV3's last conv block is a good target for visualization
        target_layer = model.Mixed_7c.branch_pool.conv
        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam(image_for_model)

        # Resize and colorize the heatmap for overlaying
        img_for_viz = cv2.resize(original_image_np, (299, 299))
        img_for_viz = np.stack([img_for_viz]*3, axis=-1)
        heatmap_resized = cv2.resize(heatmap, (299, 299))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_for_viz, 0.6, heatmap_color, 0.4, 0)

        # Create the plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(original_image_np, cmap='gray')
        axs[0].set_title(f"Original Image (28x28)\nTrue: {true_label_str}")
        axs[0].axis('off')

        axs[1].imshow(superimposed_img)
        axs[1].set_title(f"Grad-CAM Heatmap\nModel Predicted: {pred_label_str}")
        axs[1].axis('off')

        plt.tight_layout()

        # Create a descriptive filename and save the figure
        filename = f"plots/misclassified_analysis_{case_type.upper()}_idx_{case_idx}.png"
        plt.savefig(filename)
        plt.close() # Close the plot to free memory
        print(f"Analysis plot saved to {filename}")

if __name__ == "__main__":
    main()