# Self-Pruning Neural Network

## Introduction
In real-world deployments, especially in constrained environments such as UAV systems or edge devices, model efficiency is as important as accuracy.

Traditional pruning methods are applied after training, requiring additional processing and tuning. In this work, I designed a neural network that learns to prune itself during training by introducing learnable gate parameters.

---

## Methodology

### Prunable Layer
Each weight is associated with a learnable gate parameter. During forward pass:

- Gates are computed using sigmoid
- Effective weight = weight × gate

This allows smooth and differentiable pruning during training.

---

### Sparsity Regularization
To encourage pruning, an L1 penalty is applied on gate values:

Total Loss = CrossEntropy + λ × Σ(gates)

L1 regularization promotes sparsity by pushing gate values toward zero.

---

## Training Strategy
- Dataset: CIFAR-10
- Optimizer: Adam
- Gates and weights trained jointly
- Multiple λ values tested to observe trade-off

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.0001 | ~81         | ~18         |
| 0.001  | ~78         | ~42         |
| 0.01   | ~72         | ~68         |

---

## Analysis
- Low λ → high accuracy, low sparsity  
- High λ → aggressive pruning, lower accuracy  
- Optimal λ balances both  

The model naturally learns redundancy and suppresses less useful connections.

---

## Visualization
The distribution of gate values shows:
- A concentration near zero (pruned weights)
- A cluster away from zero (important weights)

---

## Conclusion
This approach demonstrates that pruning can be embedded directly into the learning process, eliminating the need for post-training pruning steps.

It is especially useful for real-time and resource-constrained systems.

---

## Future Improvements
- Structured pruning (neurons instead of weights)
- Integration with convolutional networks
- Deployment optimization for edge devices
