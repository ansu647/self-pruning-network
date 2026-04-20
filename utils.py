import torch

def compute_sparsity(model, threshold=1e-2):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < threshold).sum().item()

    return 100 * zero / total


def compute_l1_loss(model):
    l1_loss = 0
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            l1_loss += gates.sum()
    return l1_loss


def collect_all_gates(model):
    all_gates = []
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            all_gates.append(gates.detach().cpu().numpy().flatten())
    return all_gates