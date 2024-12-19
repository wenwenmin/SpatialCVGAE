# SpatialCVGAE
![image](https://github.com/wenwenmin/SpatialCVGAE/blob/main/SpatialCVGAE_Overview.jpg)
## News
2024.09.18 SpatialCVGAE based on pyG (PyTorch Geometric) framework is availble at [SpatialCVGAE](https://github.com/wenwenmin/SpatialCVGAE/blob/main/SpatialCVGAE.py).
## Overview
The advent of spatially resolved transcriptomics (SRT) has provided critical insights into the spatial context of tissue microenvironments, which is vital for understanding tissue heterogeneity. However, in SRT data, the model often suffers from instability caused by the sparsity and high noise in the data, which can be exacerbated by random initialization. To address these challenges, we propose SpatialCVGAE, a consensus clustering framework designed for SRT data. SpatialCVGAE constructs multiple spatial graphs and incorporates gene expressions of various dimensions as inputs to variational graph autoencoders (VGAEs), learning multiple latent representations for clustering. Each representation captures distinct biological characteristics or reflects different spatial structures. These clustering results are then integrated using a consensus clustering approach, which enhances the model's stability and robustness by combining multiple clustering outcomes. Experiments demonstrate that SpatialCVGAE effectively mitigates the instability caused by individual initializations, significantly enhancing both the stability and accuracy of results. Compared to traditional single-model approaches in representation learning and post-processing, this method fully leverages the diversity of multiple representations to optimize spatial domain identification, showing superior robustness and adaptability.
## Datasets
All data used in this work are available at: https://zenodo.org/records/12804689.
## Getting started
Our code has been debugged. If you want to run the code, just find run_option 1, run_option 2, run_option 3 to run the corresponding random initialization framework.
<br>For option 1, considering that random seeds affect both weight initialization and random operations during training, we assign multiple different random seeds to the same number of VGAEs for simultaneous training and use consensus clustering to mitigate the impact of randomness on the results. For option 2, we adopt a multi-view framework for training. We construct spatial graphs using different methods, and for each spatial graph, we apply the same training process as in Option 1. For option 3, we select 5,000 highly variable genes as input and then perform the same training process as in Option 1.
## Software dependencies
scanpy == 1.9.2
scipy == 1.10.1
sklearn == 1.3.1
PyG == 2.3.1
