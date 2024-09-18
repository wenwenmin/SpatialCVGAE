# SpatialCVGAE
![image](https://github.com/JinyunNiu/SpatialCVGAE/blob/main/SpatialCVGAE_Overview.jpg)
## News
2024.03.18 SpatialCVGAE based on pyG (PyTorch Geometric) framework is availble at [SpatialCVGAE](https://github.com/wenwenmin/SpatialCVGAE/blob/main/SpatialCVGAE.py).
## Overview
The development of spatially resolved transcriptomics has provided us with information on the spatial context of tissue microenvironments, which is essential for deciphering heterogeneity in tissues and mapping communication between cells. Comprehensively analyzing spatial transcriptomics data with gene expression data and spatial location information remains a challenge. In deep learning, weight initialization may have a significant impact on the results. Based on the above two problems, we propose SpatialCVGAE, which uses multiple variational graph autoencoders with different initializations to simultaneously generate corresponding low-dimensional embeddings, and then utilizes consensus clustering to integrate the clustering results of multiple low-dimensional embeddings. We find that using SpatialCVGAE can not only improve the performance of spatial domain identification task, but also smooth the impact of different initialization methods. We test SpatialCVGAE on two ST datasets and it outperform existing methods. In addition, we also demonstrate the capabilities of SpatialCVGAE inference and  denoise.
## Datasets
We will release the dataset link after the paper is reviewed.
## Getting started
Our code has been debugged. If you want to run the code, just find run_option 1, run_option 2, run_option 3 to run the corresponding random initialization framework.
<br>For option 1, considering that random seeds affect both weight initialization and random operations during training, we assign multiple different random seeds to the same number of VGAEs for simultaneous training and use consensus clustering to mitigate the impact of randomness on the results. For option 2, we adopt a multi-view framework for training. We construct spatial graphs using different methods, and for each spatial graph, we apply the same training process as in Option 1. For option 3, we randomly select 3,000 highly variable genes as input each time from an initial pool of 5,000 highly variable genes and then perform the same training process as in Option 1.
## Software dependencies
scanpy == 1.9.2
scipy == 1.10.1
sklearn == 1.3.1
PyG == 2.3.1
