# SEM-Net: Efficient Pixel Modelling for image inpainting with Spatially Enhanced SSM (WACV 2025)
========================================================================================

Image inpainting aims to repair a partially damaged image based on the information from known regions of the images. \revise{Achieving semantically plausible inpainting results is particularly challenging because it requires the reconstructed regions to exhibit similar patterns to the semanticly consistent regions}. This requires a model with a strong capacity to capture long-range dependencies. Existing models struggle in this regard due to the slow growth of receptive field for Convolutional Neural Networks (CNNs) based methods and patch-level interactions in Transformer-based methods, which are ineffective for capturing long-range dependencies. Motivated by this, we propose SEM-Net, a novel visual State Space model (SSM) vision network, modelling corrupted images at the pixel level while capturing long-range dependencies (LRDs) in state space, achieving a linear computational complexity. To address the inherent lack of spatial awareness in SSM, we introduce the Snake Mamba Block (SMB) and Spatially-Enhanced Feedforward Network. These innovations enable SEM-Net to outperform state-of-the-art inpainting methods on two distinct datasets, showing significant improvements in capturing LRDs and enhancement in spatial consistency. Additionally, SEM-Net achieves state-of-the-art performance on motion deblurring, demonstrating its generalizability. Our source code is available in supplementary materials.

========================================================================================
**Overview**
--------------------

## News
- [] Paper Download (coming soon)
- [] Training Code (coming soon)
- [] Pre-trained Models (coming soon)

