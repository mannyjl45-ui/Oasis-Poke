"""
Lightweight test-generation script for a trained DiT + VAE pair.

This is a simplified / self-contained version of generate.py focused on:
 - loading a trained DiT checkpoint and a trained VAE checkpoint (or a combined checkpoint),
 - reading a 4-frame sequence and corresponding 4 action vectors from an .npz dataset,
 - encoding frames with the VAE, running a single forward through the DiT, decoding,
 - saving one output frame as a PNG.

Designed to be run in Colab / local repo where the training outputs are under:
  /content/Oasis-Poke/checkpoints/model_final.pt
  /content/Oasis-Poke/checkpoints/vae_final.pt
or combined:
  /content/Oasis-Poke/checkpoints/model_vae_final.pt

Usage examples (from shell):
  python generate.py \
    --model-ckpt /content/Oasis-Poke/checkpoints/model_final.pt \
    --vae-ckpt /content/Oasis-Poke/checkpoints/vae_final.pt \
    --data-path /content/Oasis-Poke/training_data/frames_data.npz \
    --sample-index 0 \
    --out out_frame.png

Or programmatically from a notebook:
  from types import SimpleNamespace
  import generate
  args = SimpleNamespace(
      model_ckpt="/content/Oasis-Poke/checkpoints/model_final.pt",
      vae_ckpt="/content/Oasis-Poke/checkpoints/vae_final.pt",
      data_path="/content/Oasis-Poke/training_data/frames_data.npz",
      sample_index=0,
      seq_len=4,
      frame_index=0,
      dit_name="miniDit",
      vae_name="miniVit",
      scaling_factor=0.07843137255,
  )
  generate.test_generation(args)

Note: This script uses the repo's DiT_models and VAE_models registries (dit.py, vae.py).
It attempts to be tolerant about checkpoint formats:
 - combined checkpoints with {"model_state":..., "vae_state":...}
 - or model-only/state-dict files saved directly.

"""
import os
import argparse
from pprint import pprint

import numpy as np
from PIL import Image

import torch
from torch import nn
from einops import rearrange
from torch import autocast

# import registry constructors from the repo
from dit import DiT_models
from vae import VAE_models

# Defaults tuned to your Colab layout
DEFAULT_MODEL_CKPT = "/content/Oasis-Poke/checkpoints/model_final.pt"
DEFAULT_VAE_CKPT = "/content/Oasis-Poke/checkpoints/vae_final.pt"
DEFAULT_COMBINED_CKPT = "/content/Oasis-Poke/checkpoints/model_vae_final.pt"
DEFAULT_DATA = "/content/Oasis-Poke/training_data/frames_data.npz"


def load_possible_state(path, key_names=("model_state", "vae_state")):
    """
    Load a checkpoint and return the dict found under key (if present) or the top-level
    dict itself (if it looks like a state_dict).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu")
    # If this is a combined checkpoint like {"model_state":..., "vae_state":...}, return it whole
    if isinstance(ck, dict) and any(k in ck for k in key_names):
        return ck
    # If ck looks like a state dict (mapping param name -> tensor) return it under synthetic key
    if isinstance(ck, dict) and all(isinstance(v, torch.Tensor) or isinstance(v, (list, tuple)) for v in ck.values()):
        # Heuristic: treat as model state dict
        return {"model_state": ck}
    # fallback
    return {"model_state": ck}


def prepare_frames_and_actions_from_npz(npz_path, sample_index=0, seq_len=4):
    """
    Loads 4 frames and their actions from npz at sample_index.
    Supports frames shaped (N,H,W,C) (continuous frames) or (N,T,H,W,C).
    Returns:
      frames_t: torch.FloatTensor (T, C, H, W) in [0,1]
      actions_t: torch.FloatTensor (T, A)
    """
    data = np.load(npz_path, allow_pickle=True)
    if "frames" not in data or "actions" not in data:
        raise RuntimeError("NPZ must contain 'frames' and 'actions' arrays")

    frames = data["frames"]
    actions = data["actions"]

    # normalize frames to [0,1] float32
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    else:
        frames = frames.astype(np.float32)
        frames = np.clip(frames, 0.0, 1.0)

    # Possible layouts:
    # A) frames: (N, T, H, W, C)
    # B) frames: (N, H, W, C) treat as contiguous frames
    if frames.ndim == 5:
        N, T_total, H, W, C = frames.shape
        if sample_index >= N:
            raise IndexError("sample_index out of range")
        if T_total < seq_len:
            raise RuntimeError(f"Sample has only {T_total} frames, need {seq_len}")
        seq = frames[sample_index, :seq_len]  # (T, H, W, C)
        # actions
        act = actions[sample_index]
        if act.ndim == 1:
            act = np.tile(act[None, :], (seq_len, 1))
        else:
            act = act[:seq_len]
    elif frames.ndim == 4:
        N, H, W, C = frames.shape
        if sample_index + seq_len > N:
            raise IndexError("Not enough contiguous frames starting at sample_index")
        seq = frames[sample_index : sample_index + seq_len]  # (T, H, W, C)
        # actions align per-frame or per-sample
        if actions.ndim == 2 and actions.shape[0] == N:
            act = actions[sample_index : sample_index + seq_len]
        elif actions.ndim == 3 and actions.shape[0] == N:
            # actions: (N, T', A)
            act = actions[sample_index]
            if act.shape[0] >= seq_len:
                act = act[:seq_len]
            else:
                # replicate last if shorter
                act = np.tile(act[None, :], (seq_len, 1))[:seq_len]
        else:
            # fallback: if actions length equals seq_len assume per-frame actions stored elsewhere
            if actions.shape[0] >= seq_len:
                act = actions[:seq_len]
            else:
                raise RuntimeError("Unsupported actions layout in NPZ")
    else:
        raise RuntimeError("Unsupported frames array shape in NPZ")

    frames_t = torch.from_numpy(seq).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    actions_t = torch.from_numpy(act).float()  # (T, A)

    return frames_t, actions_t


def ensure_ext_cond_dim(model, ext_cond):
    """
    Ensure ext_cond has the dimension expected by model.external_cond (if Linear).
    Pads with zeros or truncates as necessary.
    ext_cond shape: (B, T, A)
    Returns ext_cond adapted.
    """
    if hasattr(model, "external_cond") and isinstance(model.external_cond, nn.Linear):
        expected = model.external_cond.in_features
        actual = ext_cond.shape[-1]
        if actual == expected:
            return ext_cond
        elif actual < expected:
            pad = torch.zeros((*ext_cond.shape[:-1], expected - actual), device=ext_cond.device, dtype=ext_cond.dtype)
            return torch.cat([ext_cond, pad], dim=-1)
        else:
            return ext_cond[..., :expected]
    return ext_cond


def test_generation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset frames/actions
    print("Loading sample frames/actions from:", args.data_path)
    frames_t, actions_t = prepare_frames_and_actions_from_npz(args.data_path, sample_index=args.sample_index, seq_len=args.seq_len)
    T, C_img, H, W = frames_t.shape
    print(f"Loaded {T} frames with image size {H}x{W}, channels={C_img}, actions shape = {tuple(actions_t.shape)}")

    # Build models
    print("Instantiating DiT:", args.dit_name)
    model = DiT_models[args.dit_name]()  # you can pass constructor args if needed
    print("Instantiating VAE:", args.vae_name)
    vae = VAE_models[args.vae_name]()

    # Load checkpoints (tolerant)
    model_loaded = None
    vae_loaded = None
    # Try combined checkpoint first if provided
    if args.combined_ckpt and os.path.exists(args.combined_ckpt):
        ck = load_possible_state(args.combined_ckpt)
        if "model_state" in ck:
            model_state = ck.get("model_state")
            try:
                model.load_state_dict(model_state, strict=False)
                model_loaded = args.combined_ckpt
                print("Loaded model state from combined checkpoint:", args.combined_ckpt)
            except Exception as e:
                print("Warning: failed to load model_state from combined ckpt:", e)
        if "vae_state" in ck:
            vae_state = ck.get("vae_state")
            try:
                vae.load_state_dict(vae_state, strict=False)
                vae_loaded = args.combined_ckpt
                print("Loaded VAE state from combined checkpoint:", args.combined_ckpt)
            except Exception as e:
                print("Warning: failed to load vae_state from combined ckpt:", e)

    # If not loaded from combined, try separate ckpts
    if model_loaded is None and args.model_ckpt and os.path.exists(args.model_ckpt):
        ck = load_possible_state(args.model_ckpt)
        model_state = ck.get("model_state", None) or ck
        try:
            model.load_state_dict(model_state, strict=False)
            model_loaded = args.model_ckpt
            print("Loaded model state from:", args.model_ckpt)
        except Exception as e:
            print("Warning: failed to load model checkpoint:", e)

    if vae_loaded is None and args.vae_ckpt and os.path.exists(args.vae_ckpt):
        ck = load_possible_state(args.vae_ckpt)
        vae_state = ck.get("vae_state", None) or ck
        try:
            vae.load_state_dict(vae_state, strict=False)
            vae_loaded = args.vae_ckpt
            print("Loaded VAE state from:", args.vae_ckpt)
        except Exception as e:
            print("Warning: failed to load vae checkpoint:", e)

    model = model.to(device).eval()
    vae = vae.to(device).eval()

    # Move data to device
    frames = frames_t.unsqueeze(0).to(device)  # (1, T, C, H, W)
    actions = actions_t.unsqueeze(0).to(device)  # (1, T, A)
    B = frames.shape[0]

    # VAE encoding
    scaling_factor = args.scaling_factor
    x_flat = rearrange(frames, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        if device.type == "cuda":
            with autocast("cuda", dtype=torch.half):
                z = vae.encode(x_flat * 2 - 1).mean * scaling_factor
        else:
            z = vae.encode(x_flat * 2 - 1).mean * scaling_factor

    # reshape latents to (B, T, C_z, h_z, w_z)
    try:
        p = vae.patch_size
        h_z = H // p
        w_z = W // p
        z = rearrange(z, "(b t) (h w) c -> b t c h w", b=B, t=T, h=h_z, w=w_z)
    except Exception:
        if z.ndim == 4:
            z = rearrange(z, "(b t) c h w -> b t c h w", b=B, t=T)
        else:
            # fallback square
            n, L = z.shape
            side = int(int(L) ** 0.5)
            z = z.view(B, T, 1, side, side)

    print("Latent z shape:", tuple(z.shape))

    # Prepare timesteps: zeros (deterministic single pass)
    t = torch.zeros((B, T), dtype=torch.long, device=device)

    # Ensure external cond dim
    ext = ensure_ext_cond_dim(model, actions)

    # Forward through model
    with torch.no_grad():
        if device.type == "cuda":
            with autocast("cuda", dtype=torch.half):
                out_z = model(z, t, ext)
        else:
            out_z = model(z, t, ext)

    print("Model output latent shape:", tuple(out_z.shape))

    # Decode output latents
    out_flat = rearrange(out_z, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        if device.type == "cuda":
            with autocast("cuda", dtype=torch.half):
                decoded = (vae.decode(out_flat / scaling_factor) + 1) / 2
        else:
            decoded = (vae.decode(out_flat / scaling_factor) + 1) / 2

    # decoded -> (B, T, H, W, C)
    decoded = decoded.clamp(0.0, 1.0)
    decoded = rearrange(decoded, "(b t) c h w -> b t h w c", b=B, t=T)
    print("Decoded pixel output shape:", tuple(decoded.shape))

    # Save requested frame index as PNG
    frame_idx = args.frame_index
    if frame_idx >= T:
        raise IndexError("frame_index out of range")
    out_frame = decoded[0, frame_idx]  # (H, W, C)
    out_img = (out_frame * 255.0).cpu().numpy().astype("uint8")

    # Convert to PIL image
    if out_img.ndim == 2:
        pil = Image.fromarray(out_img, mode="L")
    elif out_img.shape[2] == 1:
        pil = Image.fromarray(out_img[:, :, 0], mode="L").convert("RGB")
    else:
        pil = Image.fromarray(out_img[:, :, :3])

    # Optionally resize (user may want explicit 160x240)
    if args.save_size:
        target_h, target_w = args.save_size
        pil = pil.resize((target_w, target_h), resample=Image.BILINEAR)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pil.save(args.out)
    print("Saved output PNG to:", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DiT+VAE generation on 4-frame sample from dataset")
    parser.add_argument("--combined-ckpt", type=str, default=DEFAULT_COMBINED_CKPT, help="Combined ckpt with both model_state and vae_state (optional)")
    parser.add_argument("--model-ckpt", type=str, default=DEFAULT_MODEL_CKPT, help="Path to model checkpoint (model_final.pt or model_checkpoint_step_*.pt)")
    parser.add_argument("--vae-ckpt", type=str, default=DEFAULT_VAE_CKPT, help="Path to vae checkpoint (vae_final.pt or vae_checkpoint_step_*.pt)")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA, help="Path to .npz containing 'frames' and 'actions'")
    parser.add_argument("--sample-index", type=int, default=0, help="Which sample (or starting frame) to use")
    parser.add_argument("--seq-len", dest="seq_len", type=int, default=4, help="How many frames to take from dataset (default 4)")
    parser.add_argument("--frame-index", type=int, default=0, help="Which output frame (0..seq_len-1) to save")
    parser.add_argument("--out", type=str, default="out_frame.png", help="Path to save the PNG")
    parser.add_argument("--dit-name", type=str, default="miniDit", help="Registry key for DiT model constructor")
    parser.add_argument("--vae-name", type=str, default="miniVit", help="Registry key for VAE model constructor")
    parser.add_argument("--scaling-factor", type=float, default=0.07843137255, help="VAE scaling factor used in training")
    parser.add_argument("--save-size", nargs=2, type=int, default=None, help="Optional: [H W] to resize saved PNG to (H,W)")

    args = parser.parse_args()
    # print args and run
    print("run args:")
    pprint(vars(args))
    test_generation(args)