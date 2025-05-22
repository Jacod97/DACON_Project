import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import log_loss
from config import CFG, device
from model import BaseModel
from utils import seed_everything

def train_model(train_loader, val_loader, class_names):
    model = BaseModel(num_classes=len(class_names)).to(device)
    best_logloss = float('inf')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    for epoch in range(CFG['EPOCHS']):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Accuracy
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # LogLoss
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_logloss = log_loss(all_labels, all_probs, labels=list(range(len(class_names))))

        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"📦 Best model saved at epoch {epoch+1} (logloss: {val_logloss:.4f})")

    return model

if __name__ == "__main__":
    seed_everything(CFG['SEED'])
    from dataset import get_dataloaders
    
    train_root = './train'
    test_root = './test'
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(train_root, test_root)
    model = train_model(train_loader, val_loader, class_names) 