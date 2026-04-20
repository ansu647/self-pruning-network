# Self-Pruning Neural Network

## Overview
This project implements a **self-pruning neural network** that dynamically removes less important weights during training using learnable gating mechanisms.

Unlike traditional pruning (post-training), this approach integrates pruning directly into the learning process, enabling the model to adaptively optimize both performance and efficiency.

---

## Key Idea
Each weight is associated with a **trainable gate parameter**:

- Gate values ∈ [0, 1] (via sigmoid)
- Weight contribution = weight × gate
- Gates close to 0 → effectively prune the connection

---

## Architecture
- Custom `PrunableLinear` Layer
- Fully connected neural network:
  - Input → 400 → 200 → Output (10 classes)
- Activation: ReLU

---

## Loss Function
Total Loss:

Loss = CrossEntropy + λ × SparsityLoss

- CrossEntropy → classification performance  
- L1 penalty on gates → encourages sparsity  

---

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.0001 | ~81         | ~18         |
| 0.001  | ~78         | ~42         |
| 0.01   | ~72         | ~68         |

---

## Observations
- Increasing λ increases sparsity  
- Higher sparsity reduces accuracy  
- Moderate λ provides best trade-off  

---

## Output

### Gate Distribution
- Shows clear separation between important and pruned weights  
- Large spike near zero indicates successful pruning  

---

## Project Structure
self-pruning-network/
│
├── model.py
├── train.py
├── utils.py
├── config.py
├── requirements.txt
├── results/
│ ├── metrics.csv
│ └── gate_distribution.png


---

## How to Run
```bash
pip3 install -r requirements.txt
python3 train.py
```
Applications
Edge AI systems
UAV / drone-based ML models
Resource-constrained environments
Real-time inference systems
Key Learning

This project demonstrates how model efficiency can be integrated into the learning process itself rather than treated as a separate optimization step.

Author
Ansu Sinha
