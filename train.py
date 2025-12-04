# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import cfg
import time
import os

def train_model(model, train_loader, val_loader, test_loader, pe_name):
    model = model.to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    print(f"Start Training: {pe_name}")
    
    # 기록용 딕셔너리
    history = {'train_loss': [], 'val_acc': []}
    
    best_acc = 0.0
    save_path = f"checkpoint_{pe_name}.pth"

    for epoch in range(cfg.epochs):
        # 1. Train
        model.train()
        running_loss = 0.0
        start = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        
        # 2. Validation
        val_acc = evaluate(model, val_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        # 3. Checkpoint Saving
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved_msg = " [Saved Best]"
        else:
            saved_msg = ""

        print(f"Epoch [{epoch+1}/{cfg.epochs}] "
              f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%{saved_msg} | "
              f"Time: {time.time()-start:.1f}s")
    
    # 4. Final Test with Best Model
    print(f"\nTraining Finished. Loading Best Model for {pe_name}...")
    model.load_state_dict(torch.load(save_path))
    test_acc = evaluate(model, test_loader)
    
    print(f"[{pe_name.upper()}] Final Test Accuracy (Best Val Model): {test_acc:.2f}%")
    
    return history, test_acc

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total