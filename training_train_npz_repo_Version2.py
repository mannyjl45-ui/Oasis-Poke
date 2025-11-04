import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

# exact class names / APIs from the repository:
from vae import AutoencoderKL            # VAE implementation in repo: AutoencoderKL
from dit import DiT                      # DiT implementation in repo: DiT
from data.npz_dataset import NPZFrameActionDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_npz_config.json')
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return json.load(f)

def build_dataloader(cfg, split):
    if split == 'train':
        ds = NPZFrameActionDataset(cfg['train_npz'], memmap=cfg.get('memmap', True))
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers',4), pin_memory=True)
    else:
        ds = NPZFrameActionDataset(cfg['val_npz'], memmap=cfg.get('memmap', True))
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers',4), pin_memory=True)

def frames_to_latent(vae: AutoencoderKL, frames: torch.Tensor, sample_posterior: bool):
    """
    Encode frames via the repo VAE and return a latent shaped for DiT:
      - frames: (B, 3, H, W) in [0,1]
      - returns latents: (B, 1, C_latent, H_lat, W_lat) where H_lat*W_lat == seq_len produced by VAE
    Uses AutoencoderKL.encode(...) which returns a DiagonalGaussianDistribution-like object (posterior).
    """
    posterior = vae.encode(frames)  # posterior object from vae.encode
    if sample_posterior and vae.use_variational:
        z = posterior.sample()   # (B, seq_len, latent_dim)
    else:
        z = posterior.mode()     # (B, seq_len, latent_dim)
    # reshape z -> (B, 1, C, H_lat, W_lat)
    B, seq_len, latent_dim = z.shape
    seq_h, seq_w = vae.seq_h, vae.seq_w
    assert seq_h * seq_w == seq_len, "VAE sequence length mismatch"
    z = z.view(B, seq_h, seq_w, latent_dim)          # (B, H_lat, W_lat, C)
    z = z.permute(0, 3, 1, 2).unsqueeze(1)           # (B, 1, C, H_lat, W_lat)
    return z

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = build_dataloader(cfg, 'train')
    val_loader = build_dataloader(cfg, 'val')

    # Instantiate AutoencoderKL from repo configured for new image size (250x160)
    vae = AutoencoderKL(
        latent_dim=cfg.get('vae_latent_dim', 16),
        input_height=cfg['img_height'],
        input_width=cfg['img_width'],
        patch_size=cfg.get('vae_patch_size', 20),
        enc_dim=cfg.get('vae_enc_dim', 768),
        enc_depth=cfg.get('vae_enc_depth', 6),
        enc_heads=cfg.get('vae_enc_heads', 12),
        dec_dim=cfg.get('vae_dec_dim', 768),
        dec_depth=cfg.get('vae_dec_depth', 6),
        dec_heads=cfg.get('vae_dec_heads', 12),
        use_variational=cfg.get('vae_use_variational', True),
    ).to(device)

    # Instantiate DiT from repo. Configure DiT so its in_channels == latent_dim returned by VAE
    dit = DiT(
        input_h=vae.seq_h,
        input_w=vae.seq_w,
        patch_size=cfg.get('dit_patch_size', 2),
        in_channels=vae.latent_dim,             # ensure DiT input channels match VAE latent_dim
        hidden_size=cfg.get('dit_hidden_size', 1024),
        depth=cfg.get('dit_depth', 12),
        num_heads=cfg.get('dit_num_heads', 16),
        mlp_ratio=cfg.get('dit_mlp_ratio', 4.0),
        external_cond_dim=cfg.get('dit_external_cond_dim', 0),
        max_frames=cfg.get('dit_max_frames', 1),
    ).to(device)

    # Training config: freeze or train VAE?
    train_vae = cfg.get('train_vae', False)
    parameters = list(dit.parameters())
    if train_vae:
        parameters += list(vae.parameters())
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-4))
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    for epoch in range(cfg['epochs']):
        dit.train()
        if train_vae:
            vae.train()
        else:
            vae.eval()

        running_loss = 0.0
        total = 0
        for frames, actions in train_loader:
            # frames: (B, 3, H, W) in [0,1] as provided by the NPZ
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                # encode frames -> latents shaped for DiT
                with torch.set_grad_enabled(train_vae):
                    latents = frames_to_latent(vae, frames, sample_posterior=cfg.get('sample_posterior', False))
                # DiT expects x: (B, T, C, H, W) and t: (B, T)
                B = latents.shape[0]
                T = latents.shape[1]  # typically 1 for single-frame conditioning
                # prepare a dummy timestep tensor for DiT timestep embedder
                t = torch.zeros((B, T), dtype=torch.long, device=device)
                logits = dit(latents, t, external_cond=None)  # logits shape (B, T, C_out, H_out, W_out)
                out = logits  # (B, T, C, H, W)
                pooled = out.mean(dim=[1,3,4])   # mean over T, H, W -> (B, C)
                # action head (attach to dit if not present)
                if not hasattr(dit, 'action_head'):
                    dit.action_head = nn.Linear(pooled.shape[-1], actions.shape[-1]).to(device)
                action_logits = dit.action_head(pooled)  # (B, num_actions)

                loss = criterion(action_logits, actions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * frames.size(0)
            total += frames.size(0)

        train_loss = running_loss / max(1, total)

        # validation
        dit.eval()
        if train_vae:
            vae.eval()
        val_loss = 0.0
        total = 0
        exact_match_total = 0
        with torch.no_grad():
            for frames, actions in val_loader:
                frames = frames.to(device)
                actions = actions.to(device)
                latents = frames_to_latent(vae, frames, sample_posterior=False)
                B = latents.shape[0]
                T = latents.shape[1]
                t = torch.zeros((B, T), dtype=torch.long, device=device)
                logits = dit(latents, t, external_cond=None)
                out = logits
                pooled = out.mean(dim=[1,3,4])
                action_logits = dit.action_head(pooled)
                loss = criterion(action_logits, actions)
                val_loss += loss.item() * frames.size(0)
                preds = (torch.sigmoid(action_logits) >= 0.5).float()
                exact_match_total += (preds == actions).all(dim=1).sum().item()
                total += frames.size(0)

        val_loss = val_loss / max(1, total)
        exact_match = exact_match_total / max(1, total)
        print(f"Epoch {epoch+1}/{cfg['epochs']} train_loss={train_loss:.4f} val_loss={val_loss:.4f} exact_match={exact_match:.4f}")

        # checkpoint
        if exact_match > best_metric:
            best_metric = exact_match
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch+1,
                'vae_state': vae.state_dict(),
                'dit_state': dit.state_dict(),
                'action_head_state': dit.action_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg,
            }, os.path.join(cfg['checkpoint_dir'], 'best_repo.pth'))

if __name__ == '__main__':
    main()