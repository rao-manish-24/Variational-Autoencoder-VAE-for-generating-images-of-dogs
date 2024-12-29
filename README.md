# Variational-Autoencoder-VAE-for-generating-images-of-dogs
A Variational Autoencoder (VAE) is a generative model that maps input data to a latent space and reconstructs it back. The goal is to learn meaningful latent representations while ensuring that the latent space follows a standard normal distribution.

# Features
Dataset preparation: Splitting raw image data into training and testing subsets.
Variational Autoencoder implementation with modular encoder-decoder architecture.
Training pipeline with loss monitoring and checkpointing.
Visualization of training loss over epochs.
Intermediate reconstructed image generation for qualitative evaluation.


# Requirements
Dependencies
The following libraries are required to run the notebook:
Python 3.7+
PyTorch
torchvision
numpy
matplotlib
scikit-learn
To install the required packages, use:
pip install torch torchvision numpy matplotlib scikit-learn


# Dataset Preparation
Input Data
The dataset should consist of images of dogs stored in a directory (source_dir). Ensure the images are in common formats (e.g., .jpg, .png, .jpeg).
Splitting the Dataset
The dataset is split into training and testing subsets using the split_dataset function:
Parameters:


source_dir: Directory containing all dog images.
train_dir: Directory to store training images.
test_dir: Directory to store test images.
test_size: Fraction of images reserved for testing (default: 20%).
random_state: Ensures reproducibility of the split.
Output:


Training and test images are copied into their respective directories.
Example:
 Dataset split complete. 16463 training images, 4116 test images.



# Variational Autoencoder (VAE) Implementation
The VAE consists of the following components:
1. Encoder
The encoder extracts features from input images and maps them to the latent space. It outputs:

Mean (μ): Center of the latent distribution.
Log-variance (σ²): Spread of the latent distribution.
3. Latent Sampling
Latent vectors are sampled using the reparameterization trick:

z=μ+σ⋅ε,ε∼N(0,1)z = μ + σ \cdot ε, \quad ε \sim \mathcal{N}(0, 1)
4. Decoder
The decoder reconstructs images from the latent vectors.
5. VAE Class
Combines the encoder and decoder into a single model pipeline.

# Training
Setup
Device: Training is performed on a GPU (CUDA) if available.
Data Loader:
Training images are resized to 56x56 pixels.
Pixel values are normalized to the range [-1, 1].

Loss Function
The VAE loss combines two components:
Reconstruction Loss (L_recon):


Measures the pixel-wise difference between input and reconstructed images using Mean Squared Error (MSE): Lrecon=MSE(input,output)L_{recon} = \text{MSE}(\text{input}, \text{output})
KL Divergence Loss (L_KL):


Ensures the latent space follows a standard normal distribution: LKL=−0.5∑(1+log⁡(σ2)−μ2−σ2)L_{KL} = -0.5 \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)
# Total Loss:

 Ltotal=Lrecon+β⋅LKLL_{total} = L_{recon} + \beta \cdot L_{KL}

Beta (β): Weighting factor for KL divergence (default: 0.00025).
# Training Pipeline

Forward pass: Input images → Encoder → Latent Space → Decoder → Output images.
Loss calculation: Compute LreconL_{recon}, LKLL_{KL}, and LtotalL_{total}.

Backpropagation: Update weights using the Adam optimizer.
Save intermediate reconstructed images and model checkpoints.

# Outputs
Reconstructed Images:

Sample reconstructions are saved during training (outputs/reconstructed.png).
Model Checkpoints:
Saved after each epoch (models/saved_models/vae_model_epoch_{epoch}.pth).

# Visualization
The notebook generates a loss curve showing the VAE's training progress:

X-axis: Epoch number.
Y-axis: Average loss per epoch.
