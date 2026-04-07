"""
Module: resolutions.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    The Configuration Matrix (V4 - Deep Tech Integration).
    Defines strict 'Quality Profiles' to ensure consistent architecture scaling.
    
    Profiles:
    - DRAFT:  Fast training, low res (Debug/Thumbnail mode). No Fourier.
    - HD:     Standard 1080p quality. Fourier Features enabled.
    - CINEMA: 4K High Fidelity. Deep network, input injection, Fourier enabled.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True) # Immutable config prevents accidental runtime changes
class QualityProfile:
    name: str
    description: str
    
    # Data Dimensions (Height, Width)
    target_res: Optional[Tuple[int, int]]
    
    # Model Architecture
    hidden_layers: int
    hidden_features: int
    use_residual: bool
    use_input_injection: bool
    
    # Signal Init (The "Sharpness" Dial)
    omega_0: float
    
    # Deep Tech: Fourier Feature Mapping
    use_fourier: bool
    fourier_dim: int
    fourier_scale: float
    
    # Training Hyperparameters
    batch_size: int
    lr: float
    epochs: int
    warmup_epochs: int  # Critical for stability in deep networks

# --- The Registry ---

# 1. DRAFT PROFILE (480p)
# Purpose: Rapid prototyping. Checks if the code works in <1 minute.
DRAFT = QualityProfile(
    name="DRAFT",
    description="480p Debug Mode. Fast training, blurry details. No Fourier.",
    target_res=(480, 854),
    hidden_layers=3,
    hidden_features=128,
    use_residual=False,
    use_input_injection=False,
    omega_0=30.0,
    use_fourier=False,       # Lean mode
    fourier_dim=0,
    fourier_scale=1.0,
    batch_size=16384,
    lr=1e-4,
    epochs=1600,
    warmup_epochs=5
)

# 2. HD PROFILE (1080p)
# Purpose: Standard use case. Good balance of size/quality.
HD = QualityProfile(
    name="HD",
    description="1080p Standard. Step-Down Tapered Architecture.",
    target_res=(1080, 1920),
    hidden_layers=4, # (Corresponds to the 4 items in the list below)
    hidden_features=[256, 256, 256, 256], # V6 Tapered Width
    use_residual=True,
    use_input_injection=False,
    omega_0=20.0,
    use_fourier=True,        
    fourier_dim=256, # Massive 200KB reduction right here
    fourier_scale=5.0,      
    batch_size=262144,
    lr=2e-4,
    epochs=10000,
    warmup_epochs=25
)

# 3. CINEMA PROFILE (4K/Archival)
# Purpose: Archival Storage. "Lossless" perceptual quality.
# CAUTION: High VRAM usage.
CINEMA = QualityProfile(
    name="CINEMA",
    description="4K Ultra-High Fidelity. Input Injection + Fourier Features.",
    target_res=(2160, 3840),
    hidden_layers=8,
    hidden_features=512, 
    use_residual=True,
    use_input_injection=True, # Mandatory for 4K stability
    omega_0=30.0,
    use_fourier=True,         # Deep Tech Enabled
    fourier_dim=512,          # Massive frequency expansion for 4K
    fourier_scale=10.0,
    batch_size=32768,          # Smaller batch to fit in VRAM
    lr=1e-4,                  # Slow and steady
    epochs=5000,
    warmup_epochs=100         # Long warmup to prevent divergence
)

# Lookup dictionary for CLI arguments
PROFILES = {
    "draft": DRAFT,
    "hd": HD,
    "cinema": CINEMA
}

def get_config(profile_name: str) -> QualityProfile:
    """Safe retrieval of config with error handling."""
    if profile_name.lower() not in PROFILES:
        raise ValueError(f"Invalid Profile '{profile_name}'. Options: {list(PROFILES.keys())}")
    return PROFILES[profile_name.lower()]

# --- Sanity Check ---
if __name__ == "__main__":
    print(f"[CONFIG] Testing Profile Integrity (V4 - Fourier)...")
    p = get_config("cinema")
    print(f"Loaded: {p.name}")
    print(f"Architecture: {p.hidden_layers} layers x {p.hidden_features} units")
    print(f"Fourier Features: {'ENABLED' if p.use_fourier else 'DISABLED'} (Dim: {p.fourier_dim})")
    print(f"Warmup: {p.warmup_epochs} epochs")
    print(">>> SUCCESS: Config module ready.")
