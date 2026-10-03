"""
GTCRN Fine-Tuning Pipeline for Defence Speech Enhancement
- Fine-tunes from official pretrained GTCRN checkpoint
- Uses official HybridLoss (Complex spectral MSE + Magnitude MSE + SI-SNR)
- Evaluates on validation set after each epoch (loss + SI-SDR / STOI)
- Saves best checkpoints and logs metrics
- Strictly CPU or GPU configurable (CPU-safe)
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.gtcrn import GTCRN
from src.training.loss import HybridLoss
from src.data.dataset import SpeechEnhancementDataset
from src.evaluation.metrics import calculate_sisdr, calculate_stoi

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        # noisy_spec, clean_spec shape: (B, 257, T, 2)
        noisy_spec = batch["noisy_spec"].to(device)
        clean_spec = batch["clean_spec"].to(device)

        optimizer.zero_grad()
        pred_spec = model(noisy_spec)
        loss = criterion(pred_spec, clean_spec)

        loss.backward()
        # Gradient clipping for GRU stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_batches, 1)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    sisdr_scores = []
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            noisy_spec = batch["noisy_spec"].to(device)
            clean_spec = batch["clean_spec"].to(device)
            clean_wav = batch["clean_wav"].numpy()

            pred_spec = model(noisy_spec)
            loss = criterion(pred_spec, clean_spec)
            total_loss += loss.item()

            # Time-domain reconstruction for SI-SDR
            pred_complex = torch.view_as_complex(pred_spec.contiguous())
            enhanced_wav = torch.istft(
                pred_complex, 512, 256, 512,
                torch.hann_window(512).pow(0.5).to(device)
            ).cpu().numpy()

            for b in range(len(clean_wav)):
                c = clean_wav[b]
                e = enhanced_wav[b]
                min_l = min(len(c), len(e))
                s = calculate_sisdr(c[:min_l], e[:min_l])
                sisdr_scores.append(s)

    avg_loss = total_loss / max(num_batches, 1)
    avg_sisdr = float(np.mean(sisdr_scores)) if sisdr_scores else 0.0
    return avg_loss, avg_sisdr

def run_fine_tuning(config_path=None,
                     pretrained_ckpt="checkpoints/model_trained_on_dns3.tar",
                     data_dir="data",
                     output_dir="checkpoints/fine_tuned",
                     epochs=5,
                     batch_size=4,
                     lr=1e-4,
                     device_str="cpu"):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    print("=======================================================")
    print("           GTCRN FINE-TUNING EXECUTION                 ")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=======================================================")

    # 1. Load Data Manifests
    train_manifest_path = os.path.join(data_dir, "train", "manifest.json")
    val_def_manifest_path = os.path.join(data_dir, "val_defence", "manifest.json")
    val_gen_manifest_path = os.path.join(data_dir, "val_general", "manifest.json")

    if not os.path.exists(train_manifest_path):
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}. Run scripts/build_pilot_dataset.py first.")

    with open(train_manifest_path, "r") as f:
        train_pairs = json.load(f)
    with open(val_def_manifest_path, "r") as f:
        val_def_pairs = json.load(f)
    
    val_gen_pairs = []
    if os.path.exists(val_gen_manifest_path):
        with open(val_gen_manifest_path, "r") as f:
            val_gen_pairs = json.load(f)

    print(f"Loaded {len(train_pairs)} training pairs, {len(val_def_pairs)} defence val pairs, {len(val_gen_pairs)} general val pairs.")

    train_ds = SpeechEnhancementDataset(train_pairs, segment_length_s=3.0, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_def_ds = SpeechEnhancementDataset(val_def_pairs, segment_length_s=3.0, is_training=False)
    val_def_loader = DataLoader(val_def_ds, batch_size=batch_size, shuffle=False)

    val_gen_loader = None
    if val_gen_pairs:
        val_gen_ds = SpeechEnhancementDataset(val_gen_pairs, segment_length_s=3.0, is_training=False)
        val_gen_loader = DataLoader(val_gen_ds, batch_size=batch_size, shuffle=False)

    # 2. Load Model
    model = GTCRN().to(device)
    if os.path.exists(pretrained_ckpt):
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Initialized weights from pretrained checkpoint: {pretrained_ckpt}")
    else:
        print(f"Warning: Checkpoint {pretrained_ckpt} not found, starting from scratch!")

    criterion = HybridLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Evaluate pre-training baseline on validation set
    print("\nEvaluating initial checkpoint on validation sets...")
    init_def_loss, init_def_sisdr = evaluate(model, val_def_loader, criterion, device)
    init_gen_sisdr = 0.0
    if val_gen_loader:
        _, init_gen_sisdr = evaluate(model, val_gen_loader, criterion, device)
    print(f"Initial Defence SI-SDR: {init_def_sisdr:.2f} dB | General SI-SDR: {init_gen_sisdr:.2f} dB")

    history = {
        "train_loss": [],
        "val_def_loss": [],
        "val_def_sisdr": [],
        "val_gen_sisdr": [],
        "epoch_time_s": []
    }

    best_val_sisdr = init_def_sisdr
    best_ckpt_path = os.path.join(output_dir, "best_gtcrn_defence.tar")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_def_loss, val_def_sisdr = evaluate(model, val_def_loader, criterion, device)
        val_gen_sisdr = 0.0
        if val_gen_loader:
            _, val_gen_sisdr = evaluate(model, val_gen_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_def_loss"].append(val_def_loss)
        history["val_def_sisdr"].append(val_def_sisdr)
        history["val_gen_sisdr"].append(val_gen_sisdr)
        history["epoch_time_s"].append(epoch_time)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) - Train Loss: {train_loss:.4f} | Def SI-SDR: {val_def_sisdr:.2f} dB | Gen SI-SDR: {val_gen_sisdr:.2f} dB")

        # Save latest checkpoint
        latest_path = os.path.join(output_dir, f"gtcrn_epoch_{epoch}.tar")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_def_loss,
            "val_sisdr": val_def_sisdr
        }, latest_path)

        # Save best checkpoint (based on combined defence + general score)
        combined_score = val_def_sisdr + 0.5 * val_gen_sisdr
        best_combined = best_val_sisdr + 0.5 * init_gen_sisdr
        if combined_score >= best_combined:
            best_val_sisdr = val_def_sisdr
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_def_sisdr": val_def_sisdr,
                "val_gen_sisdr": val_gen_sisdr
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Def: {val_def_sisdr:.2f} dB, Gen: {val_gen_sisdr:.2f} dB) to {best_ckpt_path}")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! Best Val SI-SDR: {best_val_sisdr:.2f} dB")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTCRN Defence Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--out", type=str, default="checkpoints/fine_tuned")
    args = parser.parse_args()

    run_fine_tuning(
        pretrained_ckpt=args.ckpt,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device
    )
