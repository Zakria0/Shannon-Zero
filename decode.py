"""
Module: decode.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    The Steganographic Extraction Engine.
    Adheres strictly to Separation of Concerns:
    1. CryptoHandler: Translates human PINs to geometric keys.
    2. ModelLoader: Secure, in-memory artifact decompression.
    3. NeuralRenderer: Chunked, OOM-safe continuous image evaluation.
"""

import torch
import argparse
import hashlib
import gzip
import io
import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from src.models.siren import SirenNet
from configs.resolutions import QualityProfile, get_config

# --- 1. The Cryptographic Engine ---
class CryptoHandler:
    """Translates human-readable PINs into reproducible neural geometry."""
    def __init__(self, hash_dim: int = 32, device: str = "cpu"):
        self.hash_dim = hash_dim
        self.device = device

    def get_geometric_key(self, pin_str: str) -> torch.Tensor:
        """Hashes the string and seeds the generator to perfectly recreate the training tensor."""
        hash_obj = hashlib.sha256(pin_str.encode('utf-8'))
        seed = int.from_bytes(hash_obj.digest()[:8], 'little')
        
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        
        return torch.randn(1, self.hash_dim, generator=g, device=self.device)


# --- 2. The Artifact Engine ---
class ModelLoader:
    """Handles secure File I/O. Never writes decompressed state dictionaries to disk."""
    def __init__(self, device: str):
        self.device = device
        self.logger = logging.getLogger("ShannonZero.Decoder")

    def load_secure_artifact(self, artifact_path: Path) -> dict:
        """Decompresses the GZIP artifact directly into a RAM buffer."""
        if not artifact_path.exists():
            raise FileNotFoundError(f"[FATAL] Artifact not found at: {artifact_path}")
        
        self.logger.info(f"[-] Loading and decompressing artifact: {artifact_path.name}")
        
        with gzip.open(artifact_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            
        return torch.load(buffer, map_location=self.device, weights_only=False)


# --- 3. The Inference Pipeline ---
class NeuralRenderer:
    """Rebuilds the SIREN network and queries it to generate pixels."""
    def __init__(self, profile: QualityProfile, payload: dict, device: str):
        self.profile = profile
        self.device = device
        self.logger = logging.getLogger("ShannonZero.Decoder")
        
        # --- DE FIX: Cleanly unpack the payload ---
        architecture_kwargs = payload["architecture"]
        raw_state_dict = payload["model_state"]
        
        # --- DE FIX: We no longer need to "sniff" the weights. The payload tells us the mode! ---
        self.mode = architecture_kwargs["mode"]
        self.logger.info(f"[-] Architecture Payload Mode: {self.mode.upper()}")
        
        self.logger.info(f"[-] Reconstructing Autonomous Architecture...")
        # The network perfectly shape-shifts to match the artifact
        self.model = SirenNet(**architecture_kwargs).to(self.device)
        
        # Strip DDP prefixes if they exist (using the raw_state_dict)
        clean_state = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
        
        # FP16 Hydration Leak Protection
        first_tensor = next(iter(clean_state.values()))
        if first_tensor.dtype == torch.float16:
            self.logger.info("[-] FP16 payload detected. Optimizing VRAM allocations.")
            self.model.half()
            
        self.model.load_state_dict(clean_state)
        self.model.eval()
    def _build_coordinate_grid(self, h: int, w: int) -> torch.Tensor:
        """Recreates the exact (-1, 1) coordinate space used during training."""
        x = torch.linspace(-1, 1, steps=w)
        y = torch.linspace(-1, 1, steps=h)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    def render(self, geometric_key: torch.Tensor, security_level: int = 1, chunk_size: int = 32768) -> np.ndarray:
        """Queries the network in VRAM-safe chunks."""
        if self.profile.target_res is None:
            raise ValueError("Profile target resolution cannot be None for decoding.")
            
        H, W = self.profile.target_res
        coords = self._build_coordinate_grid(H, W)
        n_samples = coords.shape[0]
        
        self.logger.info(f"[-] Initiating render grid: {H}x{W} ({n_samples} queries)")
        self.logger.info(f"[-] Security Level routed as: {security_level}")
        
        outputs = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                for i in range(0, n_samples, chunk_size):
                    batch_coords = coords[i : i + chunk_size].to(self.device)
                    
                    batch_key = None
                    if geometric_key is not None:
                        batch_key = geometric_key.repeat(batch_coords.shape[0], 1)

                    # Core Neural Query
                    batch_out = self.model(
                        batch_coords, 
                        pin_embedding=batch_key, 
                        security_level=security_level
                    )
                    outputs.append(batch_out.cpu())
                    
                    # Progress indicator
                    if (i // chunk_size) % 10 == 0:
                        progress = min(100, (i / n_samples) * 100)
                        sys.stdout.write(f"\r    Rendering... {progress:.1f}%")
                        sys.stdout.flush()

        print("\r    Rendering... 100.0%  ")
        
        # Reconstruct the image matrix
        full_output = torch.cat(outputs, dim=0)
        img_tensor = full_output.view(H, W, 3).clamp(0, 1)
        
        # Convert to standard 8-bit image array
        return (img_tensor.numpy() * 255.0).astype(np.uint8)


# --- 4. The Orchestrator (CLI) ---
def execute_decoding():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("ShannonZero.Decoder")
    
    parser = argparse.ArgumentParser(description="Shannon-Zero: Artifact Decryption Engine")
    parser.add_argument("--artifact", type=str, required=True, help="Path to the compressed.siren.gz file")
    parser.add_argument("--pin", type=str, required=True, help="The cryptographic string used during training")
    parser.add_argument("--profile", type=str, required=True, help="The resolution profile (e.g., hd)")
    parser.add_argument("--output", type=str, default="extracted_secret.png", help="Output file name")
    parser.add_argument("--security_level", type=int, default=1, choices=[0, 1], 
                        help="0 = Decoy (Masked Frequencies), 1 = Secret (Full Frequencies)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n[>>>] STARTING STEGANOGRAPHIC DECRYPTION ({device.upper()})")
    
    try:
        # 1. Init Modules
        crypto = CryptoHandler(device=device)
        loader = ModelLoader(device=device)
        profile = get_config(args.profile)
        
        # 2. Pipeline Execution
        payload = loader.load_secure_artifact(Path(args.artifact))
        renderer = NeuralRenderer(profile=profile, payload=payload, device=device)
        
        # 3. Generate the Keys and Render
        if renderer.mode == "janus":
            if args.pin.strip() == "":
                logger.error("[FATAL] Artifact is JANUS encrypted. A --pin is required.")
                sys.exit(1)
            geometric_key = crypto.get_geometric_key(args.pin)
        else:
            geometric_key = None # Legacy mode bypasses cryptography

        img_array = renderer.render(geometric_key, security_level=args.security_level)

        # 4. Save to Disk
        img = Image.fromarray(img_array)
        img.save(args.output, quality=100)
        
        logger.info(f"[SUCCESS] Image perfectly extracted to: {args.output}\n")
        
    except Exception as e:
        logger.error(f"[FATAL] Decryption Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    execute_decoding()
