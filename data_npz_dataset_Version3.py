import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZFrameActionDataset(Dataset):
    """
    Loads an .npz with:
      - frames: (N, H, W, 3) or (N, 3, H, W) floats in [0,1]
      - actions: (N, 10) boolean or {0,1}
    Returns (frame_tensor, action_tensor):
      - frame_tensor: float32 torch.Tensor shape (3, H, W), values in [0,1]
      - action_tensor: float32 torch.Tensor shape (10,), values 0.0/1.0
    Use memmap=True for large NPZ to avoid loading whole file into RAM.
    """
    def __init__(self, npz_path, frame_key='frames', action_key='actions', memmap=False, transform=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        # load arrays with or without memmap
        self._npz = np.load(npz_path, mmap_mode='r') if memmap else np.load(npz_path, allow_pickle=False)
        if frame_key not in self._npz:
            raise KeyError(f"Frame key '{frame_key}' not found in {npz_path}. Available keys: {list(self._npz.keys())}")
        if action_key not in self._npz:
            raise KeyError(f"Action key '{action_key}' not found in {npz_path}. Available keys: {list(self._npz.keys())}")
        self.frames = self._npz[frame_key]
        self.actions = self._npz[action_key]

        if len(self.frames) != len(self.actions):
            raise ValueError("frames and actions length mismatch")

        # Validate dims
        if self.frames.ndim != 4:
            raise ValueError("frames must be 4D array (N,H,W,3) or (N,3,H,W)")
        if self.frames.shape[-1] == 3:
            self.format = 'NHWC'
        elif self.frames.shape[1] == 3:
            self.format = 'NCHW'
        else:
            raise ValueError("Cannot infer frame channel position; expected last dim == 3 or second dim == 3")

        # actions must be (N, num_actions)
        if self.actions.ndim != 2:
            raise ValueError("actions must be 2D array (N, num_actions)")
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def _get_frame_np(self, idx):
        f = self.frames[idx]
        if self.format == 'NHWC':
            # convert to C,H,W
            f = np.transpose(f, (2,0,1))
        # ensure float32 in range [0,1]
        return f.astype(np.float32)

    def __getitem__(self, idx):
        frame = self._get_frame_np(idx)          # (3,H,W) float32 in [0,1]
        action = self.actions[idx].astype(np.float32)  # (num_actions,) float32 0/1
        frame_t = torch.from_numpy(frame)
        action_t = torch.from_numpy(action)
        if self.transform:
            frame_t = self.transform(frame_t)
        return frame_t, action_t