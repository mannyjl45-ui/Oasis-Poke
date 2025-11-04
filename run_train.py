from types import SimpleNamespace
import train  # Assumes train.py is importable in PYTHONPATH or same directory

# Construct args like argparse Namespace
args = SimpleNamespace(
    data_path="/content/Oasis-Poke/training_data/frames_data.npz",               # path to .npz with 'frames' and 'actions'
    out_dir="./checkpoints",                              # checkpoint directory
    dit_name="miniDit",                                   # DiT registry key (as in DiT_models)
    vae_name="vit-l-20-shallow-encoder",                  # VAE registry key (as in VAE_models)
    epochs=5,
    batch_size=1,
    lr=1e-4,
    weight_decay=0.01,
    num_workers=4,
    seed=0,
    amp=True,                                             # enable mixed precision
    freeze_vae=True,                                      # freeze the VAE, only train DiT
    scaling_factor=0.07843137255,
    max_noise_level=1000,
    save_steps=500,
    save_epoch_freq=1,
    max_grad_norm=1.0,
)

# Call the main training entrypoint
train.main(args)
