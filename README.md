# DIM-HMPF
DIM-HMPF: A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images

https://img.shields.io/badge/License-MIT-blue.svg](LICENSE)
https://img.shields.io/badge/PyTorch-1.7.0%2B-red.svg](https://pytorch.org/)

This repository is the official PyTorch implementation of the paper: "A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images" (IEEE TGRS, 2026).

Authors: Jian Fang, He Sun, Xu Sun, Li Ni, Lianru Gao.

Code: https://github.com/Freelancefangjian/DIM-HMPF

📖 Overview

The core goal of fusing Hyperspectral Images (HSIs), Multispectral Images (MSIs), and Panchromatic (PAN) images is to inject spatial details from the MSI and PAN into the HSI to generate a high-resolution HSI (HR-HSI). Most existing methods rely on black-box deep learning architectures and rarely leverage this fundamental physical principle, resulting in limited interpretability.

To address this, we propose a novel Detail Injection-based Model for Hyperspectral, Multispectral, and Panchromatic imagery Fusion (DIM-HMPF). The framework is engineered to enrich the spatial detail of HSI through fusion with MSI and PAN modalities while maintaining high-fidelity spectral information. We utilize the physical model of detail injection for unfolding, employing deep learning to learn this process. An optimization problem based on the Proximal Gradient Descent (PGD) algorithm is solved through iterative steps, which are then unfolded into an end-to-end deep network. This design ensures each network variable has a clear physical meaning, significantly improving model interpretability and performance.

✨ Key Features & Contributions

•   Physically Interpretable Fusion Framework: Incorporates traditional detail injection models into a unified optimization formulation. The spatial detail injection mechanism is explicitly embedded into a learnable network via algorithm unfolding, enabling synergy between model-based priors and data-driven learning.

•   Multisource Detail Constraint: An optimization model with three detail fidelity terms is constructed to characterize the injection of spatial details from both MSI and PAN into the HSI.

•   Deep Unfolding Network: The iterative solution to the optimization problem is unfolded into an end-to-end deep network (DIM-HMPF), allowing model parameters to be adaptively learned from data.

•   Exploitation of Nonlinear Features: The nonlinear complementary features among HSI, MSI, and PAN are explored by replacing function transformation modules in the optimization model with Convolutional Neural Networks (CNNs), guided by the optimization framework.

🎯 Methodology

DIM-HMPF is derived from the traditional detail injection (e.g., BDSD) framework. The core idea is formulated as:
X = H + \Upsilon(P_D, M_D)
where X is the target HR-HSI, H is the upsampled LR-HSI, P_D and M_D are the spatial details extracted from PAN and MSI, and \(\Upsilon(\cdot)\) is a nonlinear transformation function to be learned.

An objective function with fidelity terms for PAN details, MSI details, and the final reconstruction is constructed (Eq. 12-13 in the paper). This optimization problem is solved using a Half-Quadratic Splitting (HQS) approach and optimized via iterative gradient descent (Eq. 16-17).

Network Design (See Figure 2 & 3 in the paper):
The iterative optimization steps are unfolded into a deep network with T stages (default T=5). Each stage updates four variables with clear physical meanings:
1.  \mathcal{P}_D^t: PAN detail tensor.
2.  \mathcal{M}_D^t: MSI detail tensor.
3.  \mathcal{D}^t: Fused detail tensor for injection.
4.  \mathcal{X}^t: Estimated HR-HSI.

The nonlinear function \(\Upsilon(\cdot)\) is implemented by a CNN-based proximal operator, allowing the network to learn complex feature interactions.

📊 Performance Summary

Comprehensive experiments on three public remote sensing HSI datasets (Chikusei, Houston, Xiongan) demonstrate that DIM-HMPF significantly outperforms existing mainstream fusion algorithms.

Quantitative Results on the Chikusei Dataset:
Method MPSNR (dB) ↑ RMSE ↓ ERGAS ↓ SAM (°) ↓ UIQI ↑ MSSIM ↑

EXP 19.8148 38.2580 31.8133 7.8233 0.2659 0.5415

HySure 18.6587 41.2853 37.5539 10.1042 0.3897 0.5501

D-UNet 21.2458 29.9014 12.3685 17.1766 0.1201 0.2876

HyperPNN 29.8852 9.0432 6.0435 9.3155 0.5573 0.8316

HMPNet 30.4856 8.6232 5.5363 7.9084 0.5995 0.8263

DIM-HMPF (Ours) 30.7108 8.5030 5.4223 7.5218 0.6298 0.8370

Note: DIM-HMPF achieves the best or second-best performance across all six metrics on all three datasets. For complete results (Houston, Xiongan) and visual comparisons, please refer to Section IV and Tables I-III in the paper.

🚀 Quick Start

1. Environment Setup

The code is implemented in PyTorch. The experiments in the paper were conducted on an NVIDIA RTX 3060 GPU.
# Create and activate a conda environment (recommended)
conda create -n dim-hmpf python=3.8
conda activate dim-hmpf

# Install PyTorch (Please choose the correct version for your CUDA from https://pytorch.org/)
# Example for CUDA 11.3:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install numpy scipy matplotlib tqdm scikit-image


2. Installation & Data Preparation

1.  Clone the repository
    git clone https://github.com/Freelancefangjian/DIM-HMPF.git
    cd DIM-HMPF
    
2.  Prepare Datasets
    ◦   Download the benchmark datasets:

        ▪   http://naotoyokoya.com/Download.html

        ▪   https://hyperspectral.ee.uh.edu/?page_id=1075

        ▪   Please refer to the paper for data access

    ◦   Organize the data. We provide scripts to preprocess the data and generate the simulated LR-HSI, HR-MSI, and PAN images following the protocol described in the paper (Section IV-A).
    # Modify the paths and parameters inside the script first
    python data/prepare_chikusei.py
    

3. Training

Train the DIM-HMPF model on the Chikusei dataset (default: 5 iteration stages).
python train.py --dataset Chikusei --data_path ./data/Chikusei/train --save_dir ./checkpoints --epochs 1000 --lr 1e-4

Key Training Details (from paper Section IV):
•   Loss Function: L1 loss between the estimated HR-HSI and the reference HR-HSI (Eq. 19).

•   Optimizer: Adam.

•   Batch Size: Configured based on GPU memory.

4. Test & Evaluation

Evaluate a trained model on the test set. The script will compute MPSNR, RMSE, ERGAS, SAM, UIQI, and MSSIM.
python test.py --dataset Chikusei --data_path ./data/Chikusei/test --model_path ./checkpoints/best_model.pth


📁 Project Structure


DIM-HMPF/
├── data/                   # Data loading and preprocessing utilities
│   ├── prepare_*.py       # Scripts to prepare different datasets
│   └── datasets.py        # PyTorch Dataset classes
├── models/                # Network architecture
│   ├── dim_hmpf.py       # Main DIM-HMPF model definition
│   └── proximal.py       # CNN-based proximal operator (Υ function)
├── utils/
│   ├── metrics.py         # Evaluation metrics (MPSNR, SAM, ERGAS, etc.)
│   └── logger.py          # Training logger
├── configs/               # Configuration files
├── checkpoints/           # Directory for saving trained models
├── results/               # Directory for saving test outputs
├── train.py              # Main training script
├── test.py               # Main testing and evaluation script
├── requirements.txt      # Python dependencies
└── README.md             # This file


📜 Citation

If you find this code or our paper useful for your research, please cite our paper:

```bibtex
@article{fang2026dimhmpf,
  title={A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images},
  author={Fang, Jian and Sun, He and Sun, Xu and Ni, Li and Gao, Lianru},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={64},
  pages={1--16},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TGRS.2026.3683056}
}
```

📄 License

This project is open-sourced under the MIT License. See the LICENSE file for details.

⁉️ Contact

For any questions or discussions regarding the paper or code, feel free to open an issue or contact:
•   Jian Fang: fjian1214@163.com

•   He Sun (Corresponding Author): sunhe@aircas.ac.cn

Acknowledgments: This work was supported in part by the National Natural Science Foundation of China under Grant 42325104, Grant 62571514, and Grant 62301534; and in part by the Science and Disruptive Technology Program, Aerospace Information Research Institute, Chinese Academy of Sciences (AIRCAS) under Grant 2025-AIRCAS-SDPT-17.
