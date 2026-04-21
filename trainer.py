import torch
from tqdm import tqdm
from datetime import datetime
import os
import copy
import pickle
import matplotlib.pyplot as plt
import uuid



SESSION_ID = str(uuid.uuid4())[:8]

class Trainer:
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 epochs, 
                 criterion, 
                 optimizer,
                 scheduler = None,
                 save_path: str = "./checkpoints",
                 model_name: str = "model.pth",
                 best_metric: float = float('inf')):
        
        self.model = model
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.model_name = model_name
        self.model.to(self.device)
        os.makedirs(self.save_path, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

        self.best_metric = best_metric

    def train(self, patience: int = 10):
        best_val_loss = self.best_metric
        best_model_wts = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)


            tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Patience: {patience - epochs_no_improve}, Best Val Loss: {best_val_loss:.4f}")

            if val_loss < best_val_loss:
                epochs_no_improve = 0
                tqdm.write(f"\nValidation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(best_model_wts, epoch, best_val_loss)
            else:
                epochs_no_improve += 1
            self.save_history(epoch, best_val_loss)
            if self.scheduler:
                self.scheduler.step()
                tqdm.write(f"Learning rate: {self.scheduler.get_last_lr()[0]}")

            if epochs_no_improve >= patience:
                tqdm.write(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break
        
        self.model.load_state_dict(best_model_wts)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        # Create GradScaler for mixed precision training
        scaler = torch.amp.GradScaler()
        
        for images, masks in tqdm(self.train_dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with torch.amp.autocast("cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        return epoch_loss

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Use autocast for mixed precision
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        return epoch_loss
    
    def save_model(self, model_state, epoch, metric: float):
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{SESSION_ID}_epoch_{epoch+1}_metric_{metric:.4f}_{self.model_name}"
        save_filepath = os.path.join(self.save_path, self.model_name, date_str, SESSION_ID, f"{filename}.pth")
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        torch.save(model_state, save_filepath)
        print(f"Model saved to {save_filepath}")
    
    def save_history(self, epoch, best_val_loss):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }

        date_str = datetime.now().strftime("%Y%m%d")
        history_filename = f"{SESSION_ID}_training_history_{self.model_name.split('.')[0]}_{date_str}.pkl"
        history_filepath = os.path.join(self.save_path, self.model_name, date_str, SESSION_ID, history_filename)
        os.makedirs(os.path.dirname(history_filepath), exist_ok=True)

        try:
            with open(history_filepath, 'wb') as f:
                pickle.dump(history, f)
            tqdm.write(f"Training history saved to {history_filepath}")
        except Exception as e:
            tqdm.write(f"Failed to save training history: {e}")

    
    def plot_losses(self):
        date_str = datetime.now().strftime("%Y%m%d")
        lowest_val_loss = min(self.val_losses)
        figure_name = f"loss_curve_lowest_val_{lowest_val_loss:.4f}.png"
        os.makedirs(os.path.join(self.save_path, date_str), exist_ok=True)

        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve - Lowest Val Loss: {lowest_val_loss:.4f}')
        plt.savefig(os.path.join(self.save_path, date_str, figure_name))
        plt.show()
        plt.pause(3)
        plt.close()