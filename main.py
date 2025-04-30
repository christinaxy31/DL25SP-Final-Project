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


def train_jepa(model, dataloader, device, num_epochs=20, lr=2e-4, alpha=1.0):
    """
    Train JEPA model with both global rollout loss and spatial predictor loss.
    Args:
        model: JEPAAgent
        dataloader: training dataloader
        device: torch.device
        num_epochs: number of training epochs
        lr: learning rate
        alpha: weight for spatial loss
    """
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_global_loss = 0.0
        total_spatial_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            states = batch.states.to(device)    # [B, T, 2, 64, 64]
            actions = batch.actions.to(device)  # [B, T-1, 2]

            # === Global loss ===
            preds = model(states, actions)                     # [B, T, D]
            target_repr = model.projector(                    # [B, D]
                model.encoder_backbone(states[:, -1])
            )
            global_loss = F.mse_loss(preds[:, -1], target_repr)

            # === Spatial loss ===
            feat_map, pred_map = model.forward_spatial(states[:, 0])  # [B, 64, 8, 8]
            spatial_loss = F.mse_loss(pred_map, feat_map.detach())

            # === Total loss ===
            loss = global_loss + alpha * spatial_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_global_loss += global_loss.item()
            total_spatial_loss += spatial_loss.item()
            num_batches += 1

        avg_global_loss = total_global_loss / num_batches
        avg_spatial_loss = total_spatial_loss / num_batches
        print(f"[Epoch {epoch+1}] Global Loss: {avg_global_loss:.6f}, Spatial Loss: {avg_spatial_loss:.6f}")

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
