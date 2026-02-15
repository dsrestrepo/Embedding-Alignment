import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from .classifiers import count_parameters, test_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import imageio.v2 as imageio

class AutoAlignModel(nn.Module):
    def __init__(self, base_model, gap_vector, initial_lambda=0.0):
        super(AutoAlignModel, self).__init__()
        self.base_model = base_model
        # gap_vector should be (embedding_dim,)
        # We invoke it as (1, dim) for broadcasting
        self.register_buffer('gap_vector', torch.tensor(gap_vector, dtype=torch.float32).unsqueeze(0))
        self.lambda_align = nn.Parameter(torch.tensor(initial_lambda, dtype=torch.float32))

    def forward(self, text, image):
        # Shift embeddings
        shift = (self.lambda_align / 2.0) * self.gap_vector
        text_shifted = text + shift
        image_shifted = image - shift
        
        # Normalize (L2) to unit sphere as in original code
        text_shifted = F.normalize(text_shifted, p=2, dim=1)
        image_shifted = F.normalize(image_shifted, p=2, dim=1)
        
        return self.base_model(text_shifted, image_shifted)

    def get_lambda(self):
        return self.lambda_align.item()

def train_auto_align(model_class, train_loader, test_loader, text_input_size, image_input_size, output_size, gap_vector,
                     num_epochs=5, multilabel=True, report=False, lr=3e-4, set_weights=True, adam=False, p=0.0, V=True, 
                     val_loader=None, patience=5, hidden=[128], 
                     viz_data=None, output_dir=None, dataset_name="dataset"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Initialize base model
    if 'Late' in model_class.__name__:
        base_model = model_class(text_input_size, image_input_size, output_size, p=p, hidden_images=hidden, hidden_text=hidden)
    else:
        # EarlyFusionModel typically takes 'hidden'
        base_model = model_class(text_input_size, image_input_size, output_size, p=p, hidden=hidden)
        
    model = AutoAlignModel(base_model, gap_vector, initial_lambda=0.0)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)
    
    print(f'The number of parameters of the model (including auto-align) are: {count_parameters(model)}')
    
    # Loss setup
    if set_weights:
        # Try to access labels from dataset
        if hasattr(train_loader.dataset, 'labels') and train_loader.dataset.labels is not None:
            labels_tensor = train_loader.dataset.labels
        else:
             all_labels = []
             for batch in train_loader:
                 all_labels.append(batch['labels'])
             labels_tensor = torch.cat(all_labels)

        if not multilabel:
            if labels_tensor.ndim == 1:
                class_indices = labels_tensor.cpu().numpy().astype(int)
            else:
                class_indices = torch.argmax(labels_tensor, dim=1).cpu().numpy()
            
            classes = np.unique(class_indices)
            if len(classes) > 0:
                class_weights = compute_class_weight('balanced', classes=classes, y=class_indices)
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                class_weights = None
        else:
            class_counts = labels_tensor.sum(dim=0)
            total_samples = len(labels_tensor)
            num_classes = labels_tensor.shape[1]
            class_weights = total_samples / (num_classes * torch.clamp(class_counts, min=1))
            class_weights = class_weights.float()
    else:
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        if class_weights is not None and len(class_weights) == 2:
            pos_weight = class_weights[1] / class_weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer includes lambda
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
    # Visualization setup
    viz_images_paths = []
    pca = None
    if viz_data is not None and output_dir is not None:
        try:
            text_viz, image_viz = viz_data
            # Sample if too large for PCA Viz
            max_viz_points = 5000
            if len(text_viz) > max_viz_points:
                indices = np.random.choice(len(text_viz), max_viz_points, replace=False)
                text_viz = text_viz[indices]
                image_viz = image_viz[indices]
            
            # Fit PCA once on initial data (unshifted or mean-centered)
            pca = PCA(n_components=2)
            all_emb = np.concatenate([text_viz, image_viz], axis=0)
            pca.fit(all_emb)
            print("PCA fitted for visualization.")
        except Exception as e:
            print(f"Viz setup failed: {e}")
            viz_data = None # Disable viz

        def save_viz_frame(epoch, current_lambda):
            try:
                # Calculate shifted embeddings using numpy
                shift = (current_lambda / 2.0) * gap_vector
                t_shifted = text_viz + shift
                i_shifted = image_viz - shift
                
                # Normalize
                t_shifted = normalize(t_shifted, axis=1, norm='l2')
                i_shifted = normalize(i_shifted, axis=1, norm='l2')
                
                # Transform
                all_s = np.concatenate([t_shifted, i_shifted], axis=0)
                reduced = pca.transform(all_s)
                
                red_t = reduced[:len(t_shifted)]
                red_i = reduced[len(t_shifted):]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(red_t[:, 0], red_t[:, 1], label='Text', alpha=0.5, s=10)
                plt.scatter(red_i[:, 0], red_i[:, 1], label='Image', alpha=0.5, s=10)
                plt.legend()
                plt.title(f"{dataset_name} Alignment - Epoch {epoch} - Lambda: {current_lambda:.4f}")
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                
                # Let's use auto scale for now.
                
                frame_path = os.path.join(output_dir, f"frame_{epoch:03d}.png")
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(frame_path)
                plt.close()
                return frame_path
            except Exception as e:
                print(f"Error saving viz frame: {e}")
                return None
            
        if viz_data is not None:
            path = save_viz_frame(0, 0.0)
            if path: viz_images_paths.append(path)

    # Early stopping setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    lambda_history = []
    
    # Store history for plots
    loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(text, image)
            
            if output_size == 1 and labels.ndim == 1:
                labels = labels.unsqueeze(1)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * text.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

        curr_lambda = model.module.get_lambda() if isinstance(model, nn.DataParallel) else model.get_lambda()
        lambda_history.append(curr_lambda)
    
        # Validation
        if val_loader:
             model.eval()
             val_loss = 0.0
             with torch.no_grad():
                 for batch in val_loader:
                     text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                     outputs = model(text, image)
                     
                     if output_size == 1 and labels.ndim == 1:
                        labels = labels.unsqueeze(1)
                        
                     loss = criterion(outputs, labels)
                     val_loss += loss.item() * text.size(0)
             
             val_loss = val_loss / len(val_loader.dataset)
             val_loss_history.append(val_loss)

             print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Lambda: {curr_lambda:.4f}")
             
             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 best_model_wts = copy.deepcopy(model.state_dict())
                 epochs_no_improve = 0
                 best_epoch = epoch + 1
             else:
                 epochs_no_improve += 1
                 if epochs_no_improve >= patience:
                     print(f"Early stop at {epoch + 1}")
                     if viz_data and output_dir:
                        path = save_viz_frame(epoch+1, curr_lambda)
                        if path: viz_images_paths.append(path)
                     break
        else:
             print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Lambda: {curr_lambda:.4f}")
             best_model_wts = copy.deepcopy(model.state_dict())
             best_epoch = epoch + 1

        if viz_data and output_dir:
            path = save_viz_frame(epoch+1, curr_lambda)
            if path: viz_images_paths.append(path)

    # Load best weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        
    # Generate GIF
    if viz_images_paths and output_dir:
        gif_path = os.path.join(output_dir, f"{dataset_name}_alignment_evolution.gif")
        try:
            with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
                for filename in viz_images_paths:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            print(f"GIF saved to {gif_path}")
        except Exception as e:
            print(f"Failed to create GIF: {e}")

    # Plot Lambda Evolution
    if output_dir:
        plt.figure()
        plt.plot(lambda_history)
        plt.title('Lambda Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Lambda')
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_lambda_evolution.png"))
        plt.close()

    # Final Evaluation
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        # Test Loop
        for batch in test_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            outputs = model(text, image)
            if multilabel or (output_size == 1):
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
                    
        if multilabel or (output_size == 1):
            y_pred_one_hot = (y_pred > 0.5).astype(int)
        else:
            predicted_class_indices = np.argmax(y_pred, axis=1)
            y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
        test_accuracy = accuracy_score(y_true, y_pred_one_hot)
        f1 = f1_score(y_true, y_pred_one_hot, average='macro')
        
        # AUC logic
        if multilabel or (output_size == 1):
            if output_size == 1:
                auc_scores = roc_auc_score(y_true, y_pred, average=None)
            else:
                try:
                    auc_scores = roc_auc_score(y_true, y_pred, average=None, multi_class='ovr')
                except ValueError:
                    auc_scores = np.nan
        else:
            try:
                auc_scores = roc_auc_score(y_true, y_pred_one_hot, average=None, multi_class='ovr')
            except ValueError:
                auc_scores = np.nan
        
        if output_size == 1:
            macro_auc = auc_scores
        else:    
            macro_auc = np.nanmean(auc_scores)

    final_lambda = model.module.get_lambda() if isinstance(model, nn.DataParallel) else model.get_lambda()
    print(f"Final Best Lambda (Epoch {best_epoch}): {final_lambda:.4f}")
    
    if V:
        print(f"Final Test Evaluation (Best Epoch {best_epoch}) - Accuracy: {test_accuracy:.4f}, Macro-F1: {f1:.4f}, Macro-AUC: {macro_auc:.4f}, Lambda: {final_lambda:.4f}")

    best = {
        'Acc': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores.tolist() if isinstance(auc_scores, np.ndarray) else auc_scores},
        'Macro-F1': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores.tolist() if isinstance(auc_scores, np.ndarray) else auc_scores},
        'AUC': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores.tolist() if isinstance(auc_scores, np.ndarray) else auc_scores},
        'Best_Lambda': final_lambda,
        'Lambda_History': lambda_history
    }
            
    if report:
        accuracy, precision, recall, f1 = test_model(y_true, y_pred_one_hot, V)
        return accuracy, precision, recall, f1, best
    else:
        return test_accuracy, None, None, f1, best
