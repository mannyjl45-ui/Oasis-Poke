"""
Training script for an OASIS-style DiT + ViT-VAE model on Pokemon gameplay frames.

This file is based on generate.py from the Oasis-Poke repository and adapted
to train a model (DiT + VAE) using an .npz dataset containing:
  - frames: uint8 or float32 array shaped (N, H, W, C) or (N, T, H, W, C)
  - actions: boolean or int array shaped (N, T, A) or (N, A) for per-frame actions

Goal (as requested):
  - Train on ~10k frames at resolution 250x160
  - Action vector per frame of length 10 (boolean / one-hot-like)
  - Save periodic checkpoints and a final model.

Notes / assumptions:
  - The repository provides DiT_models and VAE_models the same way generate.py does.
  - The VAE encoder/decoder interfaces are similar to generate.py.
  - This script keeps training simple (MSE loss on predicted noise / diffusion objective).
  - Hyperparameters (batch size, lr, epochs) can be tuned.
  - The script does not implement distributed training.
  - Make sure your machine has enough VRAM for batch_size and chosen model.
  - It's the user's responsibility to ensure dataset shapes match; some flexibility is built-in.

Usage:
  python train.py --data-path path/to/dataset.npz --out-dir checkpoints --epochs 10 ...

The dataset .npz should contain at least "frames" and "actions" arrays. If the frames
array is 4D (N,H,W,C) it will be treated as single-frame sequences (T=1). If it's
5D (N,T,H,W,C) it will be used as sequences. Actions should have shape (N,T,A) or (N,A).

"""

import os
import argparse
from pprint import pprint
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import functional as TF

# Import models from the repo (same as generate.py)
from dit import DiT_models
from vae import VAE_models
from utils import sigmoid_beta_schedule

# -------------------------
# Dataset
# -------------------------
class NPZSequenceDataset(Dataset):
    """
    Dataset that loads frames and actions from an .npz file.

    Expected arrays in npz:
      - frames: shape (N, H, W, C) or (N, T, H, W, C)
        dtype uint8 (0-255) or float32 (0-1).
      - actions: shape (N, A) or (N, T, A), dtype bool/int/float.

    This dataset returns:
      - frames: float tensor in [0,1] shaped (T, C, H, W)
      - actions: float tensor shaped (T, A)
    """
    def __init__(self, npz_path, seq_len=None, transform=None):
        data = np.load(npz_path, allow_pickle=True)
        if "frames" not in data or "actions" not in data:
            raise ValueError("npz must contain 'frames' and 'actions' arrays")

        frames = data["frames"]  # N,H,W,C or N,T,H,W,C
        actions = data["actions"]  # N,A or N,T,A

        # Normalize frames to float32 in [0,1]
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
        else:
            frames = frames.astype(np.float32)
            frames = np.clip(frames, 0.0, 1.0)

        # Ensure consistent dims: make frames shape (N, T, H, W, C)
        if frames.ndim == 4:
            frames = frames[:, None, ...]
        if actions.ndim == 2:
            # actions per sample: expand to per-frame if needed
            actions = actions[:, None, :]
        if actions.ndim != 3:
            raise ValueError("actions must be shape (N, A) or (N, T, A)")

        if frames.shape[0] != actions.shape[0]:
            raise ValueError("frames and actions must have same leading dimension (N)")

        self.frames = frames
        self.actions = actions

        self.N, self.T, self.H, self.W, self.C = frames.shape
        _, aT, self.A = actions.shape
        if aT != self.T:
            # if actions length differs from frames, broadcast/truncate/pad
            if aT == 1:
                actions = np.repeat(actions, self.T, axis=1)
                self.actions = actions
            else:
                # truncate or pad
                minT = min(aT, self.T)
                self.frames = self.frames[:, :minT]
                self.actions = self.actions[:, :minT]
                self.N, self.T, self.H, self.W, self.C = self.frames.shape

        self.transform = transform
        # optional sequence length to sample (for longer sequences)
        self.seq_len = seq_len if seq_len is None else int(seq_len)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return full sequence for now (T frames)
        frames = self.frames[idx]  # (T,H,W,C)
        actions = self.actions[idx]  # (T,A)

        # convert to torch tensors: frames -> (T,C,H,W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        actions = torch.from_numpy(actions).float()

        # optional transforms (random flips, crops) - keep simple
        if self.transform:
            frames = self.transform(frames)

        return frames, actions

# -------------------------
# Training utilities
# -------------------------
def collate_fn(batch):
    # Batch of (frames, actions) where frames: (T,C,H,W)
    frames = [b[0] for b in batch]
    actions = [b[1] for b in batch]
    # Stack along batch dimension: (B,T,C,H,W) and (B,T,A)
    frames = torch.stack(frames, dim=0)
    actions = torch.stack(actions, dim=0)
    return frames, actions

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# -------------------------
# Main training loop
# -------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    print("Loading dataset:", args.data_path)
    dataset = NPZSequenceDataset(args.data_path)
    print(f"Dataset: N={len(dataset)}, T={dataset.T}, H={dataset.H}, W={dataset.W}, C={dataset.C}, A={dataset.A}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Instantiate models (same constructors as generate.py)
    print("Initializing DiT model:", args.dit_name)
    model = DiT_models[args.dit_name]()
    print("Initializing VAE model:", args.vae_name)
    vae = VAE_models[args.vae_name]()

    model = model.to(device)
    vae = vae.to(device)

    # Put models in training mode (but freeze VAE optionally)
    model.train()
    if args.freeze_vae:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    else:
        vae.train()

    # optimizer: only DiT params if VAE frozen; else both
    trainable_params = list(model.parameters())
    if not args.freeze_vae:
        trainable_params += [p for p in vae.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # diffusion schedule
    max_noise_level = args.max_noise_level
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
    # expand for indexing: (T,1,1,1)
    alphas_cumprod_v = alphas_cumprod.view(-1, 1, 1, 1)

    # Training loop
    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for frames, actions in pbar:
            # frames: (B, T, C, H, W) in [0,1]
            B, T, C, H, W = frames.shape
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            # Flatten temporal and batch dims for VAE encoding like in generate.py:
            # encode input frames (we assume we condition on some prompt frames if needed;
            # here training objective is to predict noise for all frames)
            # Convert frames to (-1,1) expected by VAE: x*2 -1
            frames_flat = rearrange(frames, "b t c h w -> (b t) c h w")
            with torch.no_grad() if args.freeze_vae else torch.enable_grad():
                # use mixed precision if enabled
                if args.amp:
                    with autocast():
                        z = vae.encode(frames_flat * 2 - 1).mean
                else:
                    z = vae.encode(frames_flat * 2 - 1).mean
            # scale factor (same as generate.py)
            scaling_factor = args.scaling_factor
            z = z * scaling_factor

            # reshape back to (B, T, C_z, H_z, W_z)
            # Many VAE implementations return (N, L) or (N, P, C); generate.py rearranged by patch_size:
            try:
                # Try to respect vae.patch_size if available
                psize = vae.patch_size
                H_z = H // psize
                W_z = W // psize
                z = rearrange(z, "(b t) (h w) c -> b t c h w", t=T, h=H_z, w=W_z, b=B)
            except Exception:
                # Fallback: infer shape from z
                # assume z shape is (B*T, C_z, H_z, W_z)
                if z.ndim == 4:
                    z = rearrange(z, "(b t) c h w -> b t c h w", t=T, b=B)
                else:
                    # if z is (B*T, L) -> try to reshape to (B,T, C, H_z, W_z) using square
                    n, L = z.shape
                    side = int(int(L)**0.5)
                    # attempt C=1
                    z = z.view(B, T, 1, side, side)

            # Now z: (B, T, C_z, h, w)
            # Diffusion training objective: predict noise v or x0 given noisy x
            # We'll train a simplified objective: model predicts v (velocity / noise prediction)
            # Prepare a random timestep t for each sample (int in [0, max_noise_level-1])
            t_rand = torch.randint(low=1, high=max_noise_level, size=(B,), device=device)  # avoid 0
            # Expand t to per-frame context vector (here we use same t for all frames in sequence)
            # but could sample per-frame; we'll use per-frame same t for simplicity
            t_expand = t_rand[:, None].repeat(1, T)  # (B, T)
            t_long = t_expand.long()

            # Get alpha cumprod for these timesteps per-frame (we need indexing)
            # alphas_cumprod[t] expects a 1D index; create a flattened index for all frames
            # Prepare noisy inputs x_t from z
            # Sample standard Gaussian noise eps of same shape as z
            eps = torch.randn_like(z, device=device)

            # Get alpha_cumprod for each timestep (B,T,1,1,1)
            alpha_t = alphas_cumprod_v[t_long.view(-1)].view(B, T, 1, 1, 1)

            # x_t = sqrt(alpha) * x0 + sqrt(1 - alpha) * eps
            x0 = z
            sqrt_alpha = alpha_t.sqrt()
            sqrt_1_minus_alpha = (1 - alpha_t).sqrt()
            x_t = sqrt_alpha * x0 + sqrt_1_minus_alpha * eps

            # Prepare model inputs: flatten temporal dimension with sliding-window if required by model
            # DiT in generate.py expects inputs shaped (B, frames, c, h, w) where frames <= model.max_frames.
            # We'll feed full sequences if length fits in model.max_frames, else use cropping.
            max_frames = getattr(model, "max_frames", T)
            if T > max_frames:
                # random crop window
                start = torch.randint(low=0, high=T - max_frames + 1, size=(1,)).item()
                x_model = x_t[:, start : start + max_frames]
                actions_model = actions[:, start : start + max_frames]
                t_model = t_long[:, start : start + max_frames]
            else:
                x_model = x_t
                actions_model = actions
                t_model = t_long

            # Run model (predict v = noise prediction) under autocast if amp enabled
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                # model signature in generate.py: v = model(x_curr, t, actions)
                # but generate.py used t as long vector of noise_range indices not direct integers.
                # Here we pass t_model as long ints (timesteps)
                v_pred = model(x_model, t_model, actions_model)  # expected same shape as x_model

                # Loss: MSE between predicted v and true eps (or appropriate target depending on model)
                # generate.py treats v as noise-prediction scaled; we will train MSE to eps
                loss = nn.functional.mse_loss(v_pred, eps[:, : v_pred.shape[1]])

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item(), "step": global_step})

            # Save checkpoint every save_steps
            if global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.out_dir, f"checkpoint_step_{global_step}.pt")
                save_state = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "vae_state": vae.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if args.amp else None,
                }
                torch.save(save_state, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # End epoch: optionally save
        if (epoch + 1) % args.save_epoch_freq == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_state = {
                "global_step": global_step,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vae_state": vae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if args.amp else None,
            }
            torch.save(save_state, ckpt_path)
            print(f"Saved epoch checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(args.out_dir, "model_final.pt")
    torch.save({"model_state": model.state_dict(), "vae_state": vae.state_dict()}, final_path)
    print("Training finished. Final model saved to", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT + VAE on .npz Pokemon frames dataset")

    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset .npz file containing 'frames' and 'actions'")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--dit-name", type=str, default="DiT-S/2", help="Key for DiT_models (same registry used in generate.py)")
    parser.add_argument("--vae-name", type=str, default="vit-l-20-shallow-encoder", help="Key for VAE_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--freeze-vae", action="store_true", help="Keep VAE frozen and only train DiT")
    parser.add_argument("--scaling-factor", type=float, default=0.07843137255, help="VAE scaling factor used in generate.py")
    parser.add_argument("--max-noise-level", type=int, default=1000, help="Number of diffusion noise levels")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save-epoch-freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    print("training args:")
    pprint(vars(args))
    main(args)
