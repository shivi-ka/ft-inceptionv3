import copy
import torch

def train_model(device, dataloaders, model, criterion, optimizer, num_epochs, patience):
    # Save the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    # Intialize best validation loss
    best_val_loss = float('inf')
    # Counter for early stopping
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Placeholder to accumulate loss over entire epoch
            running_loss = 0.0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # InceptionV3 returns a main output and an auxiliary output during training
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2 # As recommended in InceptionV3 paper
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate running loss for the epoch
                # Multiply by inputs.size(0) to get the total loss for this batch
                running_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloaders[phase].dataset) 
        print(f'{phase} Loss: {epoch_loss:.4f}')

        # Early stopping and saving the best model
        if phase == 'val':
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f'\nEarly stopping triggered after {patience} epochs with no improvement.')
                model.load_state_dict(best_model_wts)
                return model
    
    model.load_state_dict(best_model_wts)
    return model