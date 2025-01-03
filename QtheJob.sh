#!/bin/bash
#SBATCH --job-name=test_ntwrk_construction
#SBATCH --output=out.txt
#SBATCH --error=error.err
#SBATCH --partition=mwgpu
#SBATCH --time=3:00:00
#SBATCH --mail-user=ymishchyriak@wesleyan.edu
#SBATCH --mail-type=ALL

source /zfshomes/ymishchyriak/winter2025/AllostericPathwayAnalyzer/venv/bin/activate
source /etc/profile.d/modules.sh
module load /share/apps/CENTOS8/ohpc/software/amber/22
python3 --version
python3 /zfshomes/ymishchyriak/winter2025/AllostericPathwayAnalyzer/network_construction.py
deactivate
