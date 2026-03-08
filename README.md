# Product Aesthetic Design: Replication & Agentic Extension

**Authors:** Runming Huang & Leran Li  
**Course:** Artificial Intelligence for Business Research  
**Date:** February 2026

## 📌 Overview
This repository contains a replication and extension of the seminal work by **Burnap et al. (2023)** on quantifying product aesthetics. 
1.  **Replication:** We reproduce the VAE-GAN architecture to analyze asthetics in generative design.
2.  **Extension:** We propose a "Foundation Model Agent" workflow (in the `Trial` folder) that integrates **Stable Diffusion** (Designer), **CLIP** (Market Agent), and **Computer Vision** (Operations Agent) to automate the discovery of Pareto-efficient designs.

## 📂 File Structure
```text
.
├── full_replicate.sh                            # Main script to execute the VAE-GAN replication
├── Trial/                                       # Folder containing the Agentic Extension code
│   ├── smoke_trail.sh                           # Script to run the Agentic AI workflow
│   └── (Agent scripts and assets)
├── slides-report-video/                         # Documentation Folder
│   ├── Replication_Report.pdf                   # Full LaTeX report
│   └── DOTE_presentation__Adjusted_ (2).pdf     # Presentation slides
├── data/                                        # Adjusted from Burnap et al. (2023)
├── external/                                    # Adjusted from Burnap et al. (2023)
├── losses/                                      # Adjusted from Burnap et al. (2023)
├── models/                                      # Adjusted from Burnap et al. (2023)
├── training/                                    # Adjusted from Burnap et al. (2023)
├── utils/                                       # Adjusted from Burnap et al. (2023)
├── xxx.log                                      # log files during replication
└── README.md                                    # Project documentation
```

## 🚀 How to Run
1. Main Replication (VAE-GAN)
To execute the full replication of the Burnap et al. architecture, run the master shell script (full_replicate.sh) in the root directory.

Note: The data directory structure mirrors the original authors' implementation. Ensure your data is organized accordingly before running.B

2. Agentic Extension (Foundation Models)

The extension code, which utilizes Stable Diffusion and CLIP agents, is located in the Trial directory. See Trial/smoke_trail.sh


## 📚 References
Burnap, A., Hauser, J. R., & Timoshenko, A. (2023). Product aesthetic design: A machine learning augmentation. Marketing Science, 42(6), 1029–1056.
