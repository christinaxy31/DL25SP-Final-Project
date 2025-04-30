import torch
import os
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
from models import JEPAAgent
import torch.nn.functional as F


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    """Load training and validation dataloaders for probing."""
    data_path = "/scratch/DL25SP"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds


def train_jepa(model, dataloader, device, num_epochs=20, lr=2e-4):
    """Train JEPA model on exploratory agent data."""
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            states = batch.states.to(device)    # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]
        
            # Forward through JEPA
            pred = model(states, actions)       # [B, T, repr_dim]
        
            # Encode the target state (last step of observation)
            target_state = states[:, -1]        # [B, 2, 64, 64]
            target_repr = model.encoder(target_state)  # [B, repr_dim]
        
            loss = F.mse_loss(pred[:, -1], target_repr)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"[JEPA Training] Epoch {epoch + 1}: avg_loss = {avg_loss:.6f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/jepa_model.pt")
    print("JEPA model saved to results/jepa_model.pt")


def load_model():
    """Initialize JEPA model. You can load checkpoint here if needed."""
    model = JEPAAgent(repr_dim=256, action_emb_dim=64)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """Train a probing head and evaluate it on JEPA representations."""
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss:.6f}")


if __name__ == "__main__":
    device = get_device()
    model = load_model()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)

    # === Train JEPA agent ===
    print("Starting JEPA training...")
    train_jepa(model, probe_train_ds, device)

    # === Evaluate by training a probing head ===
    print("Starting Probing Evaluation...")
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
