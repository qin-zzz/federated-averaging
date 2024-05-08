# fl_devices.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import *

def train_client(local_model, optimizer, client_loader, global_model, local_epochs=1):
    local_model.load_state_dict(global_model.state_dict())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        for i, data in enumerate(client_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

    return local_model.state_dict(), loss.item()

def aggregate_models(global_model, client_models):
    avg_state_dict = global_model.state_dict()

    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([client_model[key] for client_model in client_models], dim=0).mean(dim=0)

    global_model.load_state_dict(avg_state_dict)
        
    return global_model

def validate(model, test_loader):
    correct = 0
    loss = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
    return correct / total, loss / total
