# Cytocipher Manuscript code

Reproduces analyses for Cytocipher manuscript.

## Installation and Setup

    conda create -n cytocipher_ms python=3.8.12
    conda activate cytocipher_ms
    pip install cytocipher
    git clone https://github.com/BradBalderson/Cytocipher_manuscript.git
    cd Cytocipher_manuscript
    jupyter notebook
    
## Quick index
Quick explanation of respository structure.

For information on individual jupyter notebooks, see [docs/index.md](https://github.com/BradBalderson/Cytocipher_manuscript/blob/main/docs/index.md).

scripts/

    X1_simulation/
        -> Code for simulated data generation and analysis.
        
    X2_e18_hypo_neurons/
        -> Code for analysis of hypothalamus E18.5 neurons, from paper:
           [B. Yaghemaian, B. Balderson, et al. (2022)](doi.org/10.1242/dev.200076)

    X3_pbmc3k/
        -> Code for automatic download and analysis of 10X PBMC 3K data.
        
    X4_pancreas/
        -> Code for automatic download and analysis of pancreatic dev. data.
        
    X5_prostate/
        -> Code for automatic download and analysis of prostate cancer data.
        
    X6_tabula_sapiens/
        -> Code with download instructions and analysis of Tabula Sapiens data.
