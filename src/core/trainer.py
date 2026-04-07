"""
Module: trainer.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    The Training Engine (V5 - Steganographic Math Upgrade).
    
    Classes:
    1. OverfitTrainer: Standard compression.
    2. JanusTrainer:   Steganographic training.
    
    DE Improvements:
    - Null-Space Projection: Replaced Chaos MSE with Gamma L1 penalty.
    - Latent Orthogonality: Added Cosine Similarity penalty between Secret and Decoy.
    - Frequency Routing: Passing security_level to SirenNet to physically mask Fourier frequencies.
    - Safe Forwarding: Upgraded to handle intermediate tuple extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
import hashlib
from torchvision.utils import save_image
from typing import Optional, Dict, Union, Tuple

# --- 1. The Deep Tech Loss Function ---
class CharbonnierLoss(nn.Module):
    """Robust L1 Loss (Differentiable)."""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

class GradientLoss(nn.Module):
    """
    First-Order Derivative Loss using Sobel Filters.
    Forces the network to learn geometric structure (edges), not just color.
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.kernel_x = kernel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 8.0
        self.kernel_y = kernel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1) / 8.0
        
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.kernel_x.device != pred.device:
            self.kernel_x = self.kernel_x.to(pred.device)
            self.kernel_y = self.kernel_y.to(pred.device)

        pred_grad_x = F.conv2d(pred, self.kernel_x, groups=3, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, groups=3, padding=1)
        
        gt_grad_x = F.conv2d(gt, self.kernel_x, groups=3, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, groups=3, padding=1)
        
        loss = self.l1(pred_grad_x, gt_grad_x) + self.l1(pred_grad_y, gt_grad_y)
        return loss

class OverfitTrainer:
    """
    Standard V1 Trainer for Compression.
    """
    def __init__(self, 
                 model: nn.Module, 
                 dataset, 
                 run_name: str = "experiment_01",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 lr: float = 1e-4,
                 max_epochs: int = 1000,
                 warmup_epochs: int = 10,
                 gradient_weight: float = 0.0,
                 config=None):
        
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.run_name = run_name
        self.max_epochs = max_epochs
        self.gradient_weight = gradient_weight
        self.config = config
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.loss_mse = nn.MSELoss()
        self.loss_char = CharbonnierLoss()
        self.loss_grad = GradientLoss() if gradient_weight > 0 else None
        
        self.loss_weights = {'mse': 1.0, 'char': 0.2}
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

        self.best_psnr = 0.0
        self.global_step = 0
        
        self.output_dir = Path(f"experiments/{run_name}")
        self._setup_directories()

    def _setup_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "visuals").mkdir(exist_ok=True)

    def _safe_forward(self, coords: torch.Tensor, pin: Optional[torch.Tensor] = None, **kwargs):
        """
        --- DE FIX: V5 Steganographic Transport ---
        Safely unpacks intermediate tensors if return_intermediates=True
        and handles 3D Patch broadcasting.
        """
        if coords.dim() == 2:
            return self.model(coords, pin_embedding=pin, **kwargs)
        
        B, N, D = coords.shape
        flat_coords = coords.view(-1, D)
        
        flat_pin = None
        if pin is not None:
            flat_pin = pin.repeat_interleave(N, dim=0)
            
        result = self.model(flat_coords, pin_embedding=flat_pin, **kwargs)
        
        # If we asked for intermediates (Janus Mode), reconstruct the tuple properly
        if kwargs.get('return_intermediates', False):
            flat_out, gammas, latent = result
            return flat_out.view(B, N, 3), gammas, latent
        else:
            return result.view(B, N, 3)

    def train_step(self, coords: torch.Tensor, rgb: torch.Tensor) -> float:
        coords, rgb = coords.to(self.device), rgb.to(self.device)
        self.optimizer.zero_grad()
       
        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            model_output = self._safe_forward(coords)
            
            l_mse = self.loss_mse(model_output, rgb)
            l_char = self.loss_char(model_output, rgb)
            total_loss = (self.loss_weights['mse'] * l_mse) + \
                         (self.loss_weights['char'] * l_char)
            
            if self.gradient_weight > 0 and model_output.dim() == 3:
                B, N, C = model_output.shape
                patch_side = int(np.sqrt(N))
                if patch_side * patch_side == N:
                    pred_img = model_output.view(B, patch_side, patch_side, 3).permute(0, 3, 1, 2)
                    gt_img = rgb.view(B, patch_side, patch_side, 3).permute(0, 3, 1, 2)
                    total_loss += self.gradient_weight * self.loss_grad(pred_img, gt_img)

            if hasattr(self.dataset, 'update_error_map'):
                with torch.no_grad():
                    per_pixel_loss = torch.abs(model_output - rgb).mean(dim=-1) 
                    H, W = self.dataset.H, self.dataset.W
                    u = (coords[..., 0] + 1) * 0.5 * (W - 1)
                    v = (coords[..., 1] + 1) * 0.5 * (H - 1)
                    c = u.round().long().clamp(0, W-1)
                    r = v.round().long().clamp(0, H-1)
                    global_indices = r * W + c
                    self.dataset.update_error_map(global_indices.view(-1), per_pixel_loss.view(-1))
        
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss.item()

    def fit(self, epochs: int = None, batch_size: int = 8192, log_interval: int = 50):
        if epochs is None: epochs = self.max_epochs
            
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=8, pin_memory=(self.device == "cuda"))
        
        total_steps = len(dataloader)
        self._print_start_banner(batch_size)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            self.model.train()
            
            for i, batch in enumerate(dataloader):
                if isinstance(batch, dict):
                    loss = self.train_step(batch)
                else:
                    loss = self.train_step(*batch)
                
                epoch_loss += loss
                self.global_step += 1
            
            self.scheduler.step()
            avg_loss = epoch_loss / total_steps
            current_psnr = self._calculate_psnr(avg_loss)
            
            if epoch % log_interval == 0 or epoch == epochs:
                self._log_progress(epoch, epochs, avg_loss, current_psnr)
                self._save_checkpoint(epoch, current_psnr)
                self._render_snapshot(epoch, current_psnr)

    def _print_start_banner(self, batch_size):
        print(f"\n[ENGINE] Starting Run: {self.run_name}")
        print(f"         Device: {self.device.upper()}")
        print(f"         Resolution: {self.dataset.H}x{self.dataset.W}")
        print(f"         Mode: COMPRESSION (V1)")
        if self.gradient_weight > 0:
            print(f"         Gradient Loss: ENABLED (Weight: {self.gradient_weight})")

    def _calculate_psnr(self, mse_loss: float) -> float:
        if mse_loss <= 1e-10: return 100.0
        return 10 * np.log10(1.0 / mse_loss)

    def _log_progress(self, epoch, total_epochs, loss, psnr):
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{total_epochs}] | Loss: {loss:.6f} | PSNR: {psnr:.2f} dB | LR: {lr:.2e}")

    def _save_checkpoint(self, epoch: int, psnr: float):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'psnr': psnr,
            # --- DE FIX: Bake adapted architecture into the intermediate file ---
            'architecture': {
                "hidden_features": self.config.hidden_features,
                "hidden_layers": self.config.hidden_layers,
                "use_residual": self.config.use_residual,
                "use_input_injection": self.config.use_input_injection,
                "first_omega_0": self.config.omega_0,
                "hidden_omega_0": self.config.omega_0,
                "use_fourier": self.config.use_fourier,
                "fourier_dim": self.config.fourier_dim,
                "fourier_scale": self.config.fourier_scale
            } if self.config else None
        }
        torch.save(state, self.output_dir / "checkpoints" / "latest.pt")
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            torch.save(state, self.output_dir / "checkpoints" / "best_psnr.pt")
            print(f"   >>> New Record! Saved best model (PSNR: {psnr:.2f} dB)")

    def _predict_in_chunks(self, coords: torch.Tensor, pin: Optional[torch.Tensor] = None, chunk_size: int = 32768, **kwargs) -> torch.Tensor:
        """Propagates **kwargs (like security_level) down to the model during rendering."""
        self.model.eval()
        outputs = []
        n_samples = coords.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                batch_coords = coords[i : i + chunk_size].to(self.device)
                batch_pin = None
                if pin is not None:
                    current_batch_size = batch_coords.shape[0]
                    batch_pin = pin.repeat(current_batch_size, 1).to(self.device)

                batch_out = self.model(batch_coords, pin_embedding=batch_pin, **kwargs)
                outputs.append(batch_out.cpu())
        return torch.cat(outputs, dim=0)

    def _render_snapshot(self, epoch: int, psnr: float):
        full_coords = self.dataset.coords
        output = self._predict_in_chunks(full_coords) 
        H, W = self.dataset.H, self.dataset.W
        img = output.view(H, W, 3).permute(2, 0, 1)
        save_image(img, self.output_dir / "visuals" / f"epoch_{epoch}_{psnr:.1f}dB.png")
        self.model.train()


class JanusTrainer(OverfitTrainer):
    """
    V2 Trainer for Steganography (V5 Physics).
    """
    def __init__(self, *args, hash_dim: int = 32, 
                 secret_pin_str: str = "198124", 
                 decoy_pin_str: str = "000000", **kwargs):
        super().__init__(*args, **kwargs)
        self.hash_dim = hash_dim
        self.secret_pin_str = secret_pin_str
        self.decoy_pin_str = decoy_pin_str
        
        # --- DE FIX: Deterministic PIN Genesis ---
        self.pin_secret = self._generate_deterministic_tensor(self.secret_pin_str)
        self.pin_decoy  = self._generate_deterministic_tensor(self.decoy_pin_str)
        
        self.w_secret = 2.0
        self.w_decoy = 1.0
        self.w_chaos = 1.0   # Boosted for Gamma Suppression
        self.w_ortho = 0.5   # New Orthogonality Weight

    def _generate_deterministic_tensor(self, pin_str: str) -> torch.Tensor:
        """Translates a human string into a permanent, reproducible geometric tensor."""
        # 1. Create a SHA-256 hash of the PIN
        hash_obj = hashlib.sha256(pin_str.encode('utf-8'))
        
        # 2. Convert the first 8 bytes of the hash into an integer seed
        seed = int.from_bytes(hash_obj.digest()[:8], 'little')
        
        # 3. Use a local generator to avoid breaking global Adam optimizer randomness
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        
        return torch.randn(1, self.hash_dim, generator=g, device=self.device)

    def _print_start_banner(self, batch_size):
        print(f"\n[ENGINE] Starting Run: {self.run_name}")
        print(f"         Device: {self.device.upper()}")
        print(f"         Mode: JANUS PROTOCOL (V5 Security Math)")
        print(f"         Losses: Secret={self.w_secret}, Decoy={self.w_decoy}, Null-Space={self.w_chaos}, Ortho={self.w_ortho}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        coords = batch['coords'].to(self.device)
        rgb_decoy = batch['rgb_decoy'].to(self.device)
        rgb_secret = batch['rgb_secret'].to(self.device)
        batch_size = coords.shape[0]

        self.optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            
            # --- PASS 1: DECOY (Security Level 0: Masks High Frequencies) ---
            batch_pin_decoy = self.pin_decoy.repeat(batch_size, 1)
            pred_decoy, _, latent_decoy = self._safe_forward(
                coords, batch_pin_decoy, security_level=0, return_intermediates=True
            )
            loss_decoy = self.loss_mse(pred_decoy, rgb_decoy)
            
            # --- PASS 2: SECRET (Security Level 1: Full Frequency Bandwidth) ---
            batch_pin_secret = self.pin_secret.repeat(batch_size, 1)
            pred_secret, _, latent_secret = self._safe_forward(
                coords, batch_pin_secret, security_level=1, return_intermediates=True
            )
            
            loss_secret = self.loss_mse(pred_secret, rgb_secret) + \
                          0.2 * self.loss_char(pred_secret, rgb_secret)
            
            if self.gradient_weight > 0 and pred_secret.dim() == 3:
                B, N, C = pred_secret.shape
                patch_side = int(np.sqrt(N))
                if patch_side * patch_side == N:
                    pred_img = pred_secret.view(B, patch_side, patch_side, 3).permute(0, 3, 1, 2)
                    gt_img = rgb_secret.view(B, patch_side, patch_side, 3).permute(0, 3, 1, 2)
                    loss_secret += self.gradient_weight * self.loss_grad(pred_img, gt_img)

            # --- DE FIX 1: Enforced Orthogonality (Latent Subspace Separation) ---
            # Penalizes the absolute cosine similarity, forcing the angle to 90 degrees (Cosine = 0)
            cos_sim = F.cosine_similarity(latent_decoy, latent_secret, dim=-1)
            loss_ortho = torch.mean(torch.abs(cos_sim))

            # --- DE FIX 2: Null-Space Projection (The New Chaos Pass) ---
            # Feeds random pins, ignores output, penalizes the FiLM gamma magnitude to zero.
            batch_pin_chaos = torch.randn(batch_size, self.hash_dim).to(self.device)
            _, gammas_chaos, _ = self._safe_forward(
                coords, batch_pin_chaos, security_level=1, return_intermediates=True
            )
            
            # L1 penalty on all gamma tensors across all layers
            loss_chaos = sum(torch.mean(torch.abs(g)) for g in gammas_chaos)

            # --- AGGREGATE ---
            total_loss = (self.w_decoy * loss_decoy) + \
                         (self.w_secret * loss_secret) + \
                         (self.w_chaos * loss_chaos) + \
                         (self.w_ortho * loss_ortho)

        self.scaler.scale(total_loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss.item()

    def _render_snapshot(self, epoch: int, psnr: float):
        """Renders BOTH realities side-by-side using the correct frequency routing."""
        full_coords = self.dataset.coords
        
        # Must pass security_level here to properly render the visual proofs!
        out_decoy = self._predict_in_chunks(full_coords, self.pin_decoy, security_level=0)
        out_secret = self._predict_in_chunks(full_coords, self.pin_secret, security_level=1)
        
        H, W = self.dataset.H, self.dataset.W
        img_decoy = out_decoy.view(H, W, 3).permute(2, 0, 1)
        img_secret = out_secret.view(H, W, 3).permute(2, 0, 1)
        
        combined = torch.cat([img_decoy, img_secret], dim=2)
        save_image(combined, self.output_dir / "visuals" / f"epoch_{epoch}_JANUS.png")
        self.model.train()

if __name__ == "__main__":
    print("[TEST] Production Trainer Module Loaded (V5 - Steganographic Math Upgraded).")