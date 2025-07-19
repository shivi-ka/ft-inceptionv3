import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import torch.nn.functional as F


# --- Grad-CAM Implementation ---
class GradCAM:
    """
    Grad-CAM class to produce heatmaps of model activations.
    Helps visualize where the model is "looking". What parts of this X-ray made the model think this was pneumonia?"
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to the target layer to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x):
        self.model.zero_grad()
        # Forward pass to get the output. We need requires_grad for the backward pass.
        output = self.model(x.requires_grad_())
        
        # Backward pass to get the gradients
        output.backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Get the activations
        activations = self.activations.detach()
        
        # Weight the channels by the gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # Apply ReLU to keep only positive contributions to the decision
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap to be between 0 and 1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()
    
# --- Helper function to find misclassified image indices ---
def find_misclassified_indices(model, dataloader, device):
    """Iterates through a dataloader and returns the indices of misclassified images."""
    model.eval()
    misclassified_indices = {'fp': [], 'fn': []}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_binary = (torch.sigmoid(outputs) >= 0.5).cpu().numpy().flatten()
            true_labels = labels.numpy().flatten()
            
            # Find mismatches in the current batch
            for j in range(len(preds_binary)):
                global_idx = i * dataloader.batch_size + j
                if preds_binary[j] != true_labels[j]:
                    if preds_binary[j] == 1: # False Positive (predicted 1, true was 0)
                        misclassified_indices['fp'].append(global_idx)
                    else: # False Negative (predicted 0, true was 1)
                        misclassified_indices['fn'].append(global_idx)

    return misclassified_indices

def evaluate_model(device, dataloaders, model, threshold): # report card.
    print("\n--- Final Evaluation on Test Set ---")

    model.eval() # Set model to evaluation mode
    all_labels = []
    all_preds_probs = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            # Convert logits to probabilities using sigmoid
            preds_probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Collect all labels and predicted probabilities
            all_labels.extend(labels.numpy().flatten())
            all_preds_probs.extend(preds_probs.flatten())

    # Convert lists to numpy arrays for metric calculation
    all_labels = np.array(all_labels)
    all_preds_probs = np.array(all_preds_probs)
    # Convert probabilities to binary predictions based on the threshold
    all_preds_binary = (all_preds_probs >= threshold).astype(int)

    # --- Calculate and Report Metrics ---
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds_binary, target_names=['Normal (0)', 'Pneumonia (1)']))

    auc_score = roc_auc_score(all_labels, all_preds_probs)
    print(f"AUC Score: {auc_score:.4f}")

    # --- Plot and SAVE Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png') # <--- SAVE THE FIGURE
    plt.close() # <--- Close the plot to free memory
    print("Confusion Matrix plot saved to confusion_matrix.png")


    # --- Plot and SAVE ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_preds_probs)
    plt.figure() # Create a new figure
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png') # <--- SAVE THE FIGURE
    plt.close() # <--- Close the plot to free memory
    print("ROC Curve plot saved to roc_curve.png")
