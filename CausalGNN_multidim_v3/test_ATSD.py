from train import train_baseline_syn
from train_causal import test_causal_syn
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
    # train_set = torch.load(save_path + "/ATSD_v4.1_train_dataset_cor0.5.pt", weights_only=False)
    # val_set = torch.load(save_path + "/ATSD_v4.1_val_dataset_cor0.5.pt", weights_only=False)
    test_set = torch.load(save_path + "/ATSD_v4.3_test_dataset_with_mask_multidim_cor0.2_modify_norm_by_device.pt", weights_only=False)
    if args.model in ["GCN"]:
        model_func = opts.get_model(args)
        precision, recall, f1, attention_results = test_causal_syn(
            test_set=test_set,
            model_func=model_func,
            args=args
        )
    elif args.model in ["CausalGCN_CAL_M_D", "CausalGCN_CAL_M", "CausalGCN_CAL", "CausalGCN_CAL_M_D_wo_season", "CausalGCN_CAL_M_Multiscale"]:
        model_func = opts.get_model(args)
        precision, recall, f1, attention_results = test_causal_syn(
            test_set=test_set,
            model_func=model_func,
            args=args
        )
        print(attention_results[0]['edge_index'].shape)
        print(max(attention_results[0]['edge_index'][0]))
        print(max(attention_results[0]['edge_index'][1]))
    else:
        assert False

if __name__ == '__main__':
    main()
