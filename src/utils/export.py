"""
Module: export.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    The Artifact Production System.
    Adheres to Strict Separation of Concerns:
    1. NeuralCompressor: Pure logic class for model optimization (CPU/GPU agnostic).
    2. ModelArchiver: IO class for file handling.
    3. Main CLI: Orchestration layer.
"""

import torch
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import gzip
import shutil

from src.models.siren import SirenNet
from configs.resolutions import PROFILES, QualityProfile

# --- 1. Pure Logic Layer (The Brain) ---
class NeuralCompressor:
    """
    Stateless logic for reducing model footprint.
    Does not know about files, paths, or profiles.
    """
    
    @staticmethod
    def compress_weights(model: torch.nn.Module) -> torch.nn.Module:
        """
        Performs in-place quantization (FP32 -> FP16).
        """
        # Convert to Half Precision (16-bit)
        # Reduces memory by 50% immediately.
        model.half()
        return model

    @staticmethod
    def extract_state(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Extracts only the weights, stripping all metadata/gradients.
        """
        return model.state_dict()


# --- 2. IO Layer (The Hands) ---
class ModelArchiver:
    """
    Handles disk operations. Knows where things live, but not how math works.
    """
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logger = logging.getLogger("ShannonZero.Export")

    def load_checkpoint(self, filename: str = "best_psnr.pt") -> Dict[str, Any]:
        path = self.run_dir / "checkpoints" / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
        self.logger.info(f"[-] Loading heavy artifact: {path.name}")
        # Map to CPU to avoid GPU memory spikes during export
        return torch.load(path, map_location="cpu", weights_only=False)

    def save_artifact(self, state_dict: Dict[str, Any], filename: str):
        """
        Saves the model and applies GZIP entropy coding for maximum reduction.
        """
        # 1. Save Raw State Dictionary
        raw_path = self.run_dir / filename
        torch.save(state_dict, raw_path)
        
        # 2. Apply Entropy Coding (GZIP)
        zipped_path = self.run_dir / f"{filename}.gz"
        self.logger.info(f"[-] Applying Entropy Coding (GZIP)...")
        
        with open(raw_path, 'rb') as f_in:
            with gzip.open(zipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 3. Cleanup (Delete the un-zipped version)
        raw_path.unlink()
        
        self.logger.info(f"[-] Artifact Finalized: {zipped_path.name}")
        return zipped_path

    def get_file_size_kb(self, path: Path) -> float:
        return path.stat().st_size / 1024.0


# --- 3. Orchestration Layer (The Manager) ---
def export_pipeline(run_name: str, profile_name: str):
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("ShannonZero.Export")

    # Paths
    run_path = Path(f"experiments/{run_name}")
    
    # 1. Initialize Components
    archiver = ModelArchiver(run_path)
    compressor = NeuralCompressor()
    
    try:
        # 2. Load Raw Data
        checkpoint = archiver.load_checkpoint()
        
        arch_kwargs = checkpoint.get('architecture', None)
        if arch_kwargs is None:
            raise ValueError("[FATAL] Artifact lacks embedded architecture payload.")
        
        # Clean state dict BEFORE model init
        clean_state_dict = {
            key.replace('_orig_mod.', ''): value 
            for key, value in checkpoint['model_state'].items()
        }
        
        mode = "janus" if any("mapping_net" in k for k in clean_state_dict.keys()) else "legacy"
        logger.info(f"[-] Auto-Detected Artifact Mode: {mode.upper()}")
        
        # Inject the mode and hash_dim into the extracted kwargs
        arch_kwargs["mode"] = mode
        arch_kwargs["hash_dim"] = 32
        
        # 3. Reconstruct Architecture with Dynamic Adapted Mode
        model = SirenNet(**arch_kwargs)
        
        # 4. Hydrate Model (Load Weights)
        model.load_state_dict(clean_state_dict)
        
        # 5. Execute Compression Logic (Standard FP16)
        logger.info("[-] Executing Quantization Protocol (FP32 -> FP16)...")
        compressed_model = compressor.compress_weights(model)
        lean_state = compressor.extract_state(compressed_model)
        
        artifact_payload = {
            "model_state": lean_state,
            "architecture": arch_kwargs # Pass it right along to the final GZIP!
        }
        
        # 6. We save the payload instead of just the weights
        output_path = archiver.save_artifact(artifact_payload, "compressed.siren")
        
        # 7. Generate Report
        original_size = archiver.get_file_size_kb(run_path / "checkpoints/best_psnr.pt")
        final_size = archiver.get_file_size_kb(output_path)
        ratio = original_size / final_size
        
        print("\n" + "="*40)
        print(f"SHANNON-ZERO EXPORT REPORT")
        print("="*40)
        print(f"Raw Checkpoint:      {original_size:.2f} KB")
        print(f"Compressed Artifact: {final_size:.2f} KB")
        print(f"Compression Ratio:   {ratio:.2f}x")
        print("="*40 + "\n")
        
    except Exception as e:
        logger.error(f"[FATAL] Export Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shannon-Zero Artifact Exporter")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--profile", type=str, required=True, help="Profile used for training")
    args = parser.parse_args()
    
    export_pipeline(args.name, args.profile)