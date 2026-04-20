import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

from model import PrunableNet
from utils import compute_sparsity, compute_l1_loss, collect_all_gates
import config

import os
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=config.BATCH_SIZE,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=config.BATCH_SIZE,
                                         shuffle=False)

criterion = nn.CrossEntropyLoss()

results = []

for lam in config.LAMBDAS:
    print("\n==============================")
    print(f"Starting training with lambda = {lam}")
    print("Applying dynamic pruning mechanism...")
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} in progress...")
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            l1_loss = compute_l1_loss(model)

            loss = ce_loss + lam * l1_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = compute_sparsity(model)

    print("\n")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Model Sparsity Achieved: {sparsity:.2f}%")
    print("Pruning effectiveness evaluated successfully.")

    results.append([lam, accuracy, sparsity])

    # Save gate distribution for best model (example: highest sparsity)
    if lam == max(config.LAMBDAS):
        gates = collect_all_gates(model)
        all_gates = [g for layer in gates for g in layer]

        plt.hist(all_gates, bins=50)
        plt.title("Gate Value Distribution")
        plt.savefig("results/gate_distribution.png")
        plt.close()

# Save results
df = pd.DataFrame(results, columns=["Lambda", "Accuracy", "Sparsity"])
df.to_csv("results/metrics.csv", index=False)

print("\nTraining Complete. Results saved in 'results'\n")