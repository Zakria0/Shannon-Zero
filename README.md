Shannon-Zero: Neural Media Compression & Steganographic Engine
Shannon-Zero is a deep-tech neural compression engine that utilizes Sinusoidal Representation Networks (SIREN) to overfit and compress media files into highly optimized neural weights. By representing images as continuous mathematical functions rather than discrete pixel grids, Shannon-Zero achieves resolution-independent rendering and advanced cryptographic capabilities.

🧠 Core Architecture & Innovations
Adaptive Profiling (Data-Aware NAS): Analyzes spatial variance (Shannon Entropy) and High-Frequency Spectral Energy (2D FFT) prior to initialization to dynamically taper network topology and Fourier dimensions.

The Janus Protocol (Steganography): A dual-reality training mode that embeds a secret image beneath a decoy. Uses Null-Space Projection, Latent Orthogonality (forcing a 90-degree cosine similarity), and deterministic geometric PIN hashing for cryptographic retrieval.

Artifact Production Pipeline: Automated FP16 Quantization and GZIP entropy coding reduce the final model footprint by up to 50% without perceptual degradation.

Dynamic Resolution & Gradient Loss: Implements Coarse-to-Fine dynamic resolution scaling and First-Order Derivative Loss (Sobel Filters) to force the network to learn geometric structure over mere color.

📂 Project Structure

shannon-zero1/
├── configs/
│   └── resolutions.py     # Quality profiles (DRAFT, HD, CINEMA)
├── src/
│   ├── core/
│   │   └── trainer.py     # Training logic (Overfit & Janus)
│   ├── data/
│   │   └── dataset.py     # Implicit Neural Representation pipelines
│   ├── models/
│   │   └── siren.py       # Tapered SIREN architecture & Fourier Mapping
│   └── utils/
│       └── export.py      # Artifact production (Quantization & Entropy Coding)
├── main.py                # Grand Central Station (CLI Orchestrator)
├── decode.py              # Steganographic extraction and chunked rendering
└── requirements.txt       # Dependencies

⚙️ Installation
Clone the repository and install the required dependencies:

git clone https://github.com/Zakria0/Shannon-Zero
cd shannon-zero1
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

🚀 Usage Guide
Shannon-Zero operates via a central CLI orchestrator. It features three built-in quality profiles:

draft: 480p Debug Mode. Rapid prototyping without Fourier Features.

hd: 1080p Standard. Tapered Step-Down Architecture.

cinema: 4K Ultra-High Fidelity. Deep network with input injection.

1. Standard Neural Compression
Compress a single image into a neural network artifact.

python main.py --mode compression \
               --image test_image.jpg \
               --profile hd \
               --name "hd_compression_test" \
               --gradient_weight 0.1

2. The Janus Protocol (Steganography)
Train the network to represent a decoy image at security_level=0, while hiding a secret image at security_level=1, locked behind a cryptographic PIN.

python main.py --mode janus \
               --image decoy.png \
               --secret secret.png \
               --profile hd \
               --name "janus_secure_vault" \
               --pin "198124"


3. Exporting the Artifact
Once training completes, use the export pipeline to extract the pure logic, quantize the weights (FP32 to FP16), and apply GZIP entropy coding.

python src/utils/export.py --name "janus_secure_vault" --profile hd

This generates a compressed.siren.gz artifact in your experiments folder.

4. Decoding and Rendering
Reconstruct the continuous image from the compressed artifact. If using a Janus-encrypted artifact, you must provide the correct PIN and specify the security level (0 for decoy, 1 for secret).

python decode.py --artifact experiments/janus_secure_vault/compressed.siren.gz \
                 --profile hd \
                 --pin "198124" \
                 --security_level 1 \
                 --output "extracted_secret.png"



