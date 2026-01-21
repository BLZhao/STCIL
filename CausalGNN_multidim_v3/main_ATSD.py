from train import train_baseline_syn
from train_causal import train_causal_syn
from opts import setup_seed
import torch
import opts
import os
import utils
import pdb
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def main():
    args = opts.parse_args()
    save_path = "/disk1/xujing.zbl/CAL/CAL/data"
    os.makedirs(save_path, exist_ok=True)
    train_set = torch.load(save_path + "/ATSD_v4.3_train_dataset_with_mask_multidim_cor0.2_modify_norm_by_device.pt", weights_only=False)
    val_set = torch.load(save_path + "/ATSD_v4.3_val_dataset_with_mask_multidim_cor0.2_modify_norm_by_device.pt", weights_only=False)
    test_set = torch.load(save_path + "/ATSD_v4.3_test_dataset_with_mask_multidim_cor0.2_modify_norm_by_device.pt", weights_only=False)
    if args.model in ["GCN"]:
        model_func = opts.get_model(args)
        train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    elif args.model in ["CausalGCN_CAL_M_D", "CausalGCN_CAL_M", "CausalGCN_CAL", "CausalGCN_CAL_M_D_wo_season", "CausalGCN_CAL_Multiscale", "CausalGCN_CAL_M_Multiscale"]:
        model_func = opts.get_model(args)
        train_causal_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    else:
        assert False

if __name__ == '__main__':
    main()