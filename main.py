"""
Module: main.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    The Grand Central Station (V7 - Adaptive Engine Integration).
    Orchestrates both the 'Compression' and 'Security' pipelines.
    
    DE Improvements:
    - Integrated AdaptiveProfiler (Data-Aware Neural Architecture Search).
    - Flaw 1 Fixed: Janus Mode now evaluates the Maximum Envelope of both realities.
    - Flaw 3 Fixed: Enforced Constant-Time Heuristic (Epochs remain fixed).
"""

import argparse
import sys
import logging
from pathlib import Path
from dataclasses import replace
import torch
import numpy as np
from PIL import Image

torch.set_float32_matmul_precision('high')

# Deep Tech Stack
from configs.resolutions import get_config, PROFILES, QualityProfile
from src.data.dataset import PixelFittingDataset, JanusDataset
from src.models.siren import SirenNet
from src.core.trainer import OverfitTrainer, JanusTrainer

# --- Mathematical Probes (The Pre-Flight Engine) ---

class AdaptiveProfiler:
    """
    Executes Data-Aware Neural Architecture Search before initialization.
    """
    @staticmethod
    def analyze_image(path: str) -> tuple[float, float]:
        img = Image.open(path).convert('L')
        img_arr = np.array(img)
        
        # Probe B: Spatial Variance (Shannon Entropy)
        hist, _ = np.histogram(img_arr, bins=256, range=(0, 256))
        p = hist / np.sum(hist)
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        H_spatial = entropy / 8.0  # Normalize against 8-bit max
        
        # Probe A: High-Frequency Spectral Energy (2D FFT)
        f = np.fft.fft2(img_arr / 255.0)
        fshift = np.fft.fftshift(f)
        
        # Parseval's Theorem (Square the magnitudes for Energy)
        power_spectrum = np.abs(fshift) ** 2
        
        h, w = img_arr.shape
        cy, cx = h // 2, w // 2
        
        # --- DE FIX: The DC Gravity Well ---
        # The center pixel (0 Hz) represents the average brightness of the image.
        # It contains 99% of the raw energy and blinds the math to actual textures.
        # We must zero it out to measure pure structural (AC) energy.
        power_spectrum[cy, cx] = 0.0
        
        r = min(h, w) * 0.15  # Mask center 15% (Low Frequencies)
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        
        low_ac_energy = np.sum(power_spectrum[mask])
        total_ac_energy = np.sum(power_spectrum)
        
        if total_ac_energy == 0:
            E_hf = 0.0
        else:
            raw_ehf = 1.0 - (low_ac_energy / total_ac_energy)
            # With the massive DC component removed, the math is clean.
            # We apply a gentle 3.0x curve to spread natural images across the 0.0 -> 1.0 dial.
            E_hf = min(1.0, raw_ehf * 3.0)
        
        return E_hf, H_spatial

    @staticmethod
    def generate_adaptive_profile(base_cfg: QualityProfile, E_hf: float, H_spatial: float, logger: logging.Logger) -> QualityProfile:
        logger.info(f"[-] Adaptive Engine: Ehf={E_hf:.2f} | H_sp={H_spatial:.2f}")
        
        # 1. Frequency Parameters (Driven by E_hf)
        f_dim = int(base_cfg.fourier_dim * max(0.2, E_hf))
        if f_dim > 0 and f_dim % 2 != 0: f_dim += 1 # Ensure even number for sin/cos pairs
        f_dim = max(16, f_dim) if base_cfg.use_fourier else 0
        
        o_0 = 15.0 if E_hf < 0.3 else base_cfg.omega_0
        f_scale = base_cfg.fourier_scale * max(0.5, E_hf)
        
        # 2. Structural Parameters (Driven by H_spatial)
        width_scale = 1.0
        if H_spatial < 0.4:
            width_scale = 0.5
        elif H_spatial < 0.7:
            width_scale = 0.75
            
        new_features = base_cfg.hidden_features
        if isinstance(new_features, list):
            new_features = [max(16, int(w * width_scale)) for w in new_features]
        else:
            new_features = max(16, int(new_features * width_scale))
            
        use_res = True if base_cfg.hidden_layers > 4 else base_cfg.use_residual
        use_inj = True if base_cfg.hidden_layers > 6 else base_cfg.use_input_injection
        
        # 3. Physics Compensator
        # Inverse LR scaling: smaller network needs higher LR.
        lr_multiplier = 1.0 / max(0.25, width_scale ** 2)
        new_lr = min(1e-3, base_cfg.lr * lr_multiplier)
        
        # Constant-Time Heuristic: Epochs strictly remain at base limit.
        new_epochs = base_cfg.epochs
        
        # Warmup boost for deep networks with high initial frequencies
        new_warmup = base_cfg.warmup_epochs
        if o_0 >= 30.0 and base_cfg.hidden_layers >= 4:
            new_warmup = max(50, base_cfg.warmup_epochs)
            
        adapted = replace(
            base_cfg,
            fourier_dim=f_dim,
            omega_0=o_0,
            fourier_scale=f_scale,
            hidden_features=new_features,
            use_residual=use_res,
            use_input_injection=use_inj,
            lr=new_lr,
            epochs=new_epochs,
            warmup_epochs=new_warmup
        )
        
        logger.info(f"    -> Tapering Width:  {new_features} (Scale: {width_scale}x)")
        logger.info(f"    -> Fourier Dim:     {f_dim} (Scale: {f_scale:.1f})")
        logger.info(f"    -> Sharpness (w0):  {o_0}")
        logger.info(f"    -> Adjusting LR:    {new_lr:.1e} (Warmup: {new_warmup})")
        
        return adapted

# --- Infrastructure Layer ---

def setup_logging(run_name: str) -> logging.Logger:
    log_dir = Path(f"experiments/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("ShannonZero")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_dir / "system.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s')) 
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def resolve_device(forced_device: str = None) -> str:
    if forced_device: return forced_device
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# --- Orchestration Layer ---

class SessionManager:
    """
    Polymorphic Session Handler.
    Decides whether to build a Compression Pipeline or a Security Pipeline.
    """
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.name)
        self.config: QualityProfile = self._load_config()
        self.device = resolve_device(args.device)
        
        # State
        self.dataset = None
        self.model = None
        self.trainer = None

    def _load_config(self) -> QualityProfile:
        try:
            cfg = get_config(self.args.profile)
            self.logger.info(f"[-] Profile Ceiling: {cfg.name} | {cfg.description}")
            return cfg
        except ValueError as e:
            self.logger.error(f"[FATAL] Config Error: {e}")
            sys.exit(1)

    def _calculate_safe_batch_size(self, patch_size: int) -> int:
        raw_batch_size = self.config.batch_size
        
        if patch_size <= 1:
            return raw_batch_size
        
        pixels_per_patch = patch_size ** 2
        safe_batch_size = raw_batch_size // pixels_per_patch
        safe_batch_size = max(1, safe_batch_size)
        
        self.logger.warning(f"[RESOURCE] Throttling Batch Size for Stability.")
        self.logger.warning(f"           Original: {raw_batch_size} items (Pixels)")
        self.logger.warning(f"           Patch Load: {pixels_per_patch}x per item")
        self.logger.warning(f"           New Batch: {safe_batch_size} items (Patches)")
        
        return safe_batch_size

    def build_pipeline(self):
        """Factory method for pipeline construction."""
        if self.args.mode == "compression":
            self._build_compression_stack()
        elif self.args.mode == "janus":
            self._build_security_stack()
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")

    def _build_compression_stack(self):
        """V1 Optimization Stack (Gradient Loss + Patches)"""
        self.logger.info("\n[0/3] Running Pre-Flight Entropy Profiler...")
        #E_hf, H_spatial = AdaptiveProfiler.analyze_image(self.args.image)
        #self.config = AdaptiveProfiler.generate_adaptive_profile(self.config, E_hf, H_spatial, self.logger)

        self.logger.info("\n[1/3] Initializing Optimization Data Pipeline...")
        patch_size = 16 if self.args.gradient_weight > 0 else 1
        safe_batch = self._calculate_safe_batch_size(patch_size)
        self.config = replace(self.config, batch_size=safe_batch)
        
        self.dataset = PixelFittingDataset(
            image_path=self.args.image, 
            target_resolution=self.config.target_res,
            patch_size=patch_size
        )
        self.logger.info(f"      Source: {self.args.image}")
        self.logger.info(f"      Patching: {patch_size}x{patch_size}")

        self.logger.info("[2/3] Constructing SIREN (Legacy Mode)...")
        self.model = SirenNet(
            hidden_features=self.config.hidden_features,
            hidden_layers=self.config.hidden_layers,
            use_residual=self.config.use_residual,
            use_input_injection=self.config.use_input_injection,
            first_omega_0=self.config.omega_0,
            hidden_omega_0=self.config.omega_0,
            mode="legacy",
            use_fourier=self.config.use_fourier,
            fourier_dim=self.config.fourier_dim,
            fourier_scale=self.config.fourier_scale
        )

        self.model = torch.compile(self.model)

        self.logger.info("[3/3] Initializing Overfit Trainer...")
        self.trainer = OverfitTrainer(
            model=self.model,
            dataset=self.dataset,
            run_name=self.args.name,
            device=self.device,
            lr=self.config.lr,
            max_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            gradient_weight=self.args.gradient_weight,
            config=self.config
        )

    def _build_security_stack(self):
        """V2 Security Stack (Janus Protocol)"""
        if not self.args.secret:
            self.logger.error("[FATAL] Janus Mode requires --secret image argument.")
            sys.exit(1)

        # DE FIX: The Maximum Envelope 
        self.logger.info("\n[0/3] Running Pre-Flight Entropy Profiler (Janus Max-Envelope)...")
        E_hf_decoy, H_sp_decoy = AdaptiveProfiler.analyze_image(self.args.image)
        E_hf_sec, H_sp_sec = AdaptiveProfiler.analyze_image(self.args.secret)
        
        self.logger.info(f"    -> Decoy metrics:  Ehf={E_hf_decoy:.2f}, H_sp={H_sp_decoy:.2f}")
        self.logger.info(f"    -> Secret metrics: Ehf={E_hf_sec:.2f}, H_sp={H_sp_sec:.2f}")
        
        E_hf = max(E_hf_decoy, E_hf_sec)
        H_spatial = max(H_sp_decoy, H_sp_sec)
        self.config = AdaptiveProfiler.generate_adaptive_profile(self.config, E_hf, H_spatial, self.logger)

        self.logger.info("\n[1/3] Initializing Dual-Reality Data Pipeline...")
        patch_size = 16
        safe_batch = self._calculate_safe_batch_size(patch_size)
        self.config = replace(self.config, batch_size=safe_batch)
            
        self.dataset = JanusDataset(
            decoy_path=self.args.image,
            secret_path=self.args.secret,
            target_resolution=self.config.target_res,
            patch_size=patch_size
        )
        
        self.logger.info("[2/3] Constructing SIREN (Janus Mode)...")
        self.model = SirenNet(
            hidden_features=self.config.hidden_features,
            hidden_layers=self.config.hidden_layers,
            use_residual=self.config.use_residual,
            use_input_injection=self.config.use_input_injection,
            first_omega_0=self.config.omega_0,
            hidden_omega_0=self.config.omega_0,
            mode="janus",
            hash_dim=32,
            use_fourier=self.config.use_fourier,
            fourier_dim=self.config.fourier_dim,
            fourier_scale=self.config.fourier_scale
        )

        self.model = torch.compile(self.model)

        self.logger.info("[3/3] Initializing Janus Trainer...")
        self.trainer = JanusTrainer(
            model=self.model,
            dataset=self.dataset,
            run_name=self.args.name,
            device=self.device,
            lr=self.config.lr,
            max_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            gradient_weight=self.args.gradient_weight,
            hash_dim=32,
            secret_pin_str=self.args.pin,
            config=self.config
        )

    def run(self):
        self.logger.info(f"\n[>>>] STARTING {self.args.mode.upper()} PROTOCOL...")
        try:
            self.trainer.fit(
                epochs=self.config.epochs, 
                batch_size=self.config.batch_size,
                log_interval=50
            )
            self.logger.info("\n[SUCCESS] Pipeline Complete.")
            self.logger.info(f"Artifacts: experiments/{self.args.name}/")
            
        except KeyboardInterrupt:
            self.logger.warning("\n[!] User Interrupt. Checkpointing...")
            self.trainer._save_checkpoint(epoch=0, psnr=0.0)
            sys.exit(0)
        except Exception as e:
            self.logger.critical(f"[CRASH] {e}", exc_info=True)
            sys.exit(1)

# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Shannon-Zero: Neural Signal Compression")
    
    # Core Args
    parser.add_argument("--mode", type=str, default="compression", choices=["compression", "janus"],
                        help="Operation Mode: 'compression' (V1) or 'janus' (V2 Security)")
    parser.add_argument("--image", type=str, required=True, help="Source image (Decoy in Janus mode)")
    parser.add_argument("--secret", type=str, default=None, help="Secret image (Required for Janus mode)")
    
    # Config Args
    parser.add_argument("--profile", type=str, default="draft", help="Quality Profile Ceiling")
    parser.add_argument("--name", type=str, default="default_run", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Compute backend override")
    parser.add_argument("--pin", type=str, default="198124", 
                        help="Cryptographic PIN for Steganographic mapping")
    
    # Optimization Args
    parser.add_argument("--gradient_weight", type=float, default=0.0, 
                        help="Weight for Sobel Gradient Loss (0.0 = Off). Rec: 0.1 for sharpness.")

    args = parser.parse_args()

    session = SessionManager(args)
    session.build_pipeline()
    session.run()

if __name__ == "__main__":
    main()