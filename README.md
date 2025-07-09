# ft-inceptionv3
Fine-Tuned Inception-V3 Model for Pneumonia Classification

```bash
$ python3 train.py
Using device: cuda
Training data shape: (3882, 28, 28)
Validation data shape: (524, 28, 28)
Test data shape: (624, 28, 28)
Class imbalance weight (for pneumonia class): 0.11

--- STAGE 1: Training the Classifier Head ---
        Training: AuxLogits.fc.weight
        Training: AuxLogits.fc.bias
        Training: fc.weight
        Training: fc.bias
Epoch 1/5
----------
val Loss: 0.1862
Epoch 2/5
----------
val Loss: 0.1216
Epoch 3/5
----------
val Loss: 0.1146
Epoch 4/5
----------
val Loss: 0.1605
Epoch 5/5
----------
val Loss: 0.1075

--- STAGE 2: Fine-Tuning the Full Model ---
Epoch 1/10
----------
val Loss: 0.0703
Epoch 2/10
----------
val Loss: 0.0612
Epoch 3/10
----------
val Loss: 0.0537
Epoch 4/10
----------
val Loss: 0.0478
Epoch 5/10
----------
val Loss: 0.0530
Epoch 6/10
----------
val Loss: 0.0407
Epoch 7/10
----------
val Loss: 0.0396
Epoch 8/10
----------
val Loss: 0.0387
Epoch 9/10
----------
val Loss: 0.0428
Epoch 10/10
----------
val Loss: 0.0414

--- Saving Model to models/best_model.pth ---
Model saved successfully.

--- Final Evaluation on Test Set ---

--- Final Evaluation on Test Set ---

Classification Report:
               precision    recall  f1-score   support

   Normal (0)       0.92      0.81      0.86       234
Pneumonia (1)       0.89      0.96      0.93       390

     accuracy                           0.90       624
    macro avg       0.91      0.89      0.89       624
 weighted avg       0.91      0.90      0.90       624

AUC Score: 0.9623
Confusion Matrix plot saved to confusion_matrix.png
ROC Curve plot saved to roc_curve.png

--- Finding a misclassified case for analysis ---
Analyzing Case Index: 23
Case Type: False Negative
True Label: Pneumonia
Model Predicted: Normal (Probability: 0.3994)
Analysis plot saved to plots/misclassified_analysis_FN_idx_23.png
```