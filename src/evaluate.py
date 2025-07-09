import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

def evaluate_model(device, dataloaders, model, threshold):
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
    # Justification for metrics:
    # - Recall is vital to minimize false negatives (missed pneumonia cases).
    # - F1-score provides a balance between recall and precision.
    # - AUC offers a robust measure of the model's class separability.
    print(classification_report(all_labels, all_preds_binary, target_names=['Normal (0)', 'Pneumonia (1)']))

    # --- Calculate AUC Score ---
    try:
        auc_score = roc_auc_score(all_labels, all_preds_probs)
        print(f"AUC Score: {auc_score:.4f}")
    except ValueError:
        # This can happen if the test set contains only one class
        auc_score = -1
        print("AUC Score could not be calculated (only one class present in labels).")

    # --- Plot Evaluation Metrics ---
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Evaluation Metrics', fontsize=16)

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
    disp.plot(cmap=plt.cm.Blues, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.grid(False) # Hide grid lines for a cleaner look

    # Plot ROC Curve, but only if AUC was calculable
    if auc_score != -1:
        fpr, tpr, _ = roc_curve(all_labels, all_preds_probs)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Dashed line for random chance
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, 'ROC Curve not available', horizontalalignment='center', verticalalignment='center')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')


    # Adjust layout to prevent titles from overlapping and display the plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()