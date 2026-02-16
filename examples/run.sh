#!/bin/bash
# Testing the training script for DFEI
python ../wmpgnn/analysis/trainer.py --config train_DFEI_b_duaghters_only.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_cpt.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_full.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoTrue.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoIP.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoDFEI.yaml
# Testing the evaluation script for DFEI
python ../wmpgnn/analysis/evaluate.py --config evaluate_DFEI.yaml

# Testing the training script for IFT
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoTrue.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoIP.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoDFEI.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoDFEI_DFEI.yaml