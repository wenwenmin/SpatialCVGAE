# SpatialCVGAE
![image](https://github.com/JinyunNiu/SpatialCVGAE/blob/main/SpatialCVGAE_Overview.jpg)
## News
2024.03.18 SpatialCVGAE based on pyG (PyTorch Geometric) framework is availble at [VGAE_model](https://github.com/JinyunNiu/SpatialCVGAE/blob/main/VGAE_model.py).
## Overview
The development of spatially resolved transcriptomics has provided us with information on the spatial context of tissue microenvironments, which is essential for deciphering heterogeneity in tissues and mapping communication between cells. Comprehensively analyzing spatial transcriptomics data with gene expression data and spatial location information remains a challenge. In deep learning, weight initialization may have a significant impact on the results. Based on the above two problems, we propose SpatialCVGAE, which uses multiple variational graph autoencoders with different initializations to simultaneously generate corresponding low-dimensional embeddings, and then utilizes consensus clustering to integrate the clustering results of multiple low-dimensional embeddings. We find that using SpatialCVGAE can not only improve the performance of spatial domain identification task, but also smooth the impact of different initialization methods. We test SpatialCVGAE on two ST datasets and it outperform existing methods. In addition, we also demonstrate the capabilities of SpatialCVGAE inference and   denoise.
## Getting started
Our code has been debugged. If you want to run DLPFC or BRCA1 datasets on SpatialACVGAE, just go to the corresponding file and run run_single.
## Software dependencies
scanpy == 1.9.2
scipy == 1.10.1
sklearn == 1.3.1
PyG == 2.3.1
