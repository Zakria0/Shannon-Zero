"""
Module: siren.py
Author: Zakaria Oulhadj (Lead Architect)
Context: Project Shannon-Zero

Description:
    Implements the Sinusoidal Representation Network (SIREN).
    
    Production Improvements (V6 - Tapered Architecture / Pre-Training Optimization):
    1. Dynamic Tapering: hidden_features now accepts a List[int] to step-down network width.
    2. Latent Anchoring: film_gen scales dynamically based on local layer width while 
       accepting a fixed-size global security key.
    3. Graceful Residuals: Residual connections automatically bypass during width transitions.
    
    Ref: 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains' (Tancik et al.)
         'Implicit Neural Representations with Periodic Activation Functions' (Sitzmann et al.)
"""

import torch
from torch import nn
import numpy as np
from typing import Optional, Tuple, List, Union

class FourierFeatureMapping(nn.Module):
    """
    The Mathematical Lens.
    Projects raw coordinates into a high-dimensional, high-frequency space.
    """
    def __init__(self, in_features: int, fourier_dim: int, fourier_scale: float = 10.0):
        super().__init__()
        # 1. Generate the Random Gaussian Matrix (B)
        B_matrix = torch.randn(fourier_dim, in_features) * fourier_scale
        
        # --- DE FIX: Spectral Sorting ---
        magnitudes = torch.norm(B_matrix, dim=-1)
        sorted_indices = torch.argsort(magnitudes)
        B_matrix = B_matrix[sorted_indices]
        
        # 2. Fuse to State Dict
        self.register_buffer('B', B_matrix)
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Math: gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)]^T
        """
        proj = (2 * np.pi * v) @ self.B.t()
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MappingNetwork(nn.Module):
    """
    The Security Key Generator.
    Maps a SHA-256 Hash to a Latent Style Code.
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.LeakyReLU(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.LeakyReLU(0.2))
            
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SineLayer(nn.Module):
    """
    Atomic Unit: Linear -> FiLM (Optional) -> Sine Activation.
    Now supports asymmetric input/output routing for Tapered Architectures.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 is_first: bool = False, 
                 omega_0: float = 30.0,
                 use_film: bool = False,      
                 latent_dim: int = 0):        
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.use_film = use_film
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if self.use_film:
            # Latent dim is the global security key size (e.g. 256)
            # out_features is the local layer width (e.g. 128)
            self.film_gen = nn.Linear(latent_dim, out_features * 2)
            
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                lim = 1 / self.in_features
                self.linear.weight.uniform_(-lim, lim)
            else:
                lim = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-lim, lim)
            
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-lim, lim)
                
            if self.use_film:
                self.film_gen.weight.normal_(0, 0.01)
                out_dim = self.linear.out_features
                self.film_gen.bias.data[:out_dim].fill_(1.0)
                self.film_gen.bias.data[out_dim:].fill_(0.0)

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pre_activation = self.linear(x)
        gamma_out = None
        
        if self.use_film and latent is not None:
            film_params = self.film_gen(latent)
            gamma, beta = film_params.chunk(2, dim=-1)
            pre_activation = gamma * pre_activation + beta
            gamma_out = gamma 
            
        return torch.sin(self.omega_0 * pre_activation), gamma_out


class SirenNet(nn.Module):
    """
    Production SIREN Architecture (V6 - Tapered Topology).
    """
    def __init__(self, 
                 in_features: int = 2, 
                 out_features: int = 3, 
                 hidden_features: Union[int, List[int]] = 256, 
                 hidden_layers: int = 5, 
                 first_omega_0: float = 30.0, 
                 hidden_omega_0: float = 30.0,
                 use_residual: bool = False,
                 use_input_injection: bool = True,
                 mode: str = "legacy",         
                 hash_dim: int = 32,
                 use_fourier: bool = False,
                 fourier_dim: int = 256,
                 fourier_scale: float = 10.0):
        super().__init__()
        
        self.use_residual = use_residual
        self.use_input_injection = use_input_injection
        self.residual_scale = 1.0 / np.sqrt(2)
        self.mode = mode
        
        self.use_film = (mode == "janus")
        
        # --- DE FIX: Tapered Architecture Support ---
        if isinstance(hidden_features, int):
            self.hidden_features_list = [hidden_features] * hidden_layers
        else:
            self.hidden_features_list = hidden_features
            
        # The mapping network must output a vector wide enough for the widest layer
        self.latent_dim = self.hidden_features_list[0] 
        
        # 1. Mapping Network (Janus Only)
        if self.use_film:
            self.mapping_net = MappingNetwork(
                in_features=hash_dim, 
                out_features=self.latent_dim
            )
            
        # 2. Fourier Feature Mapping
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.fourier_mapping = FourierFeatureMapping(
                in_features=in_features, 
                fourier_dim=fourier_dim, 
                fourier_scale=fourier_scale
            )
            network_in_features = fourier_dim * 2
        else:
            self.fourier_mapping = None
            network_in_features = in_features
        
        # 3. Input Layer
        self.net_first = SineLayer(
            network_in_features, self.hidden_features_list[0], 
            is_first=True, omega_0=first_omega_0,
            use_film=self.use_film, latent_dim=self.latent_dim
        )
        
        # 4. Hidden Layers (Dynamically sized)
        self.layers = nn.ModuleList()
        current_in_features = self.hidden_features_list[0]
        
        for i in range(len(self.hidden_features_list)):
            out_feats = self.hidden_features_list[i]
            is_injection_layer = (i > 0) and (i % 4 == 0) and use_input_injection
            layer_in_features = current_in_features + in_features if is_injection_layer else current_in_features
            
            self.layers.append(SineLayer(
                layer_in_features, out_feats, 
                is_first=False, omega_0=hidden_omega_0,
                use_film=self.use_film, latent_dim=self.latent_dim
            ))
            current_in_features = out_feats

        # 5. Output Layer
        self.final_linear = nn.Linear(current_in_features, out_features)
        self._init_final_layer(current_in_features, hidden_omega_0)

    def _init_final_layer(self, hidden_features, omega):
        with torch.no_grad():
            lim = np.sqrt(6 / hidden_features) / omega
            self.final_linear.weight.uniform_(-lim, lim)
            self.final_linear.bias.data.fill_(0.0)

    def forward(self, 
                coords: torch.Tensor, 
                pin_embedding: Optional[torch.Tensor] = None,
                security_level: int = 1,
                return_intermediates: bool = False):
        x = coords
        latent = None
        gamma_tensors = []
        
        if self.use_film and pin_embedding is not None:
            latent = self.mapping_net(pin_embedding)
        
        if self.use_fourier:
            x = self.fourier_mapping(x)
            
            if security_level == 0:
                half_dim = x.shape[-1] // 2
                mask = torch.ones_like(x)
                mask[..., half_dim:] = 0.0
                x = x * mask
        
        x, g = self.net_first(x, latent)
        if g is not None: gamma_tensors.append(g)
        
        for i, layer in enumerate(self.layers):
            previous_x = x 
            
            if layer.in_features > x.shape[-1]:
                x = torch.cat([x, coords], dim=-1)
            
            x, g = layer(x, latent)
            if g is not None: gamma_tensors.append(g)
            
            # Residual automatically skips if shape changes (e.g. 256 -> 128)
            if self.use_residual and (x.shape == previous_x.shape):
                x = (previous_x + x) * self.residual_scale
                
        out = torch.sigmoid(self.final_linear(x))
        
        if return_intermediates:
            return out, gamma_tensors, latent
        return out

# --- DE Verification Block ---
if __name__ == "__main__":
    print("[TEST] Initializing Production SIREN (V6 - Tapered Topology)...")
    
    dummy_coords = torch.randn(10, 2)
    dummy_pin = torch.randn(10, 32)
    
    # 1. Test Tapered Setup
    print("[-] Testing Step-Down Architecture [256, 256, 128, 128]...")
    tapered_model = SirenNet(mode="janus", hidden_features=[256, 256, 128, 128], use_fourier=True)
    out, gammas, lat = tapered_model(dummy_coords, pin_embedding=dummy_pin, return_intermediates=True)
    
    assert out.shape == (10, 3)
    # 1 first layer + 4 hidden layers = 5 gamma tracking tensors
    assert len(gammas) == 5 
    print("    [OK] Tapered Forward Pass Clean.")
    print(f"    [OK] Final Linear Input Dimension verified as: {tapered_model.final_linear.in_features}")
    print(">>> SUCCESS: siren.py upgraded to V6 Tapered Topology.")