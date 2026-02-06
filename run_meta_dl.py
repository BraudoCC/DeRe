import os
import sys
import torch
import torch.multiprocessing as mp
import re
import argparse
import yaml
import numpy as np
import pandas as pd
from itertools import chain
from torch import nn
from utils.myutils import Utils
from sklearn import preprocessing
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split, ConcatDataset
from meta.networks import meta_predictor, MetaICL, MetaSimpleICL, MetaICLFrozenComp, MetaICLAddComp, MetaICLLabelEncoder, MetaICLDeepInput, MetaICLTabPFN
from tqdm import tqdm
import logging
import random
import joblib
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_path(path):
    # '/data2/coding/tsgym/TSGym_benchmark/results_long_term_forecasting/resultsGym_MLP/ETTh1/LTF_TSGym1000000_False_False_DishTS_DFT_False_series-encoding_MLP_DNN_null_True_False_False_ETTh1_ftM_sl96_ll48_pl192_dm64_el2_dl1_df256_fc3_ebtimeF_dtTrue_Exp_epochs30_lfHUBER_lr0.0001_lrscosine_0'
    tsgym_name = path.split("/")[-1]
    compoents_list = tsgym_name.split("ftM")[0].split("_")[2:-1]
    # ['False', 'False', 'DishTS', 'DFT', 'False', 'series-encoding', 'MLP', 'DNN', 'null', 'True', 'False', 'False', 'ETTh1']
    gym_x_mark, gym_series_sampling, gym_series_norm, gym_series_decomp, gym_CI, gym_series_tokenizer, gym_model, gym_backbone = compoents_list[0], compoents_list[1], compoents_list[2], compoents_list[3], compoents_list[4], compoents_list[5], compoents_list[6], compoents_list[7]

    compoents_dict = {
        "gym_x_mark": gym_x_mark, 
        "gym_series_sampling": gym_series_sampling, 
        "gym_series_norm": gym_series_norm, 
        "gym_series_decomp": gym_series_decomp, 
        "gym_CI": gym_CI, 
        "gym_series_tokenizer": gym_series_tokenizer, 
        "gym_model": gym_model, 
        "gym_backbone": gym_backbone,
        "path": path  # Add path for later use
    }
    return compoents_dict

def is_PatchTST(compoents_dict):
    return compoents_dict['gym_series_tokenizer'] == 'series-patching'

def is_DLinear(compoents_dict):
    return (compoents_dict['gym_model'] == 'MLP') and \
           (compoents_dict['gym_backbone'] == 'DNN') and \
           (compoents_dict['gym_series_decomp'] == 'MA')

def is_OLinear(compoents_dict):
    return (compoents_dict['gym_model'] == 'MLP') and \
           (compoents_dict['gym_backbone'] == 'NormLin') and \
           ("ortho" in compoents_dict['gym_series_tokenizer'])

def is_autoformer(compoents_dict):
    return (compoents_dict['gym_model'] == 'Transformer') and \
           ("auto" in compoents_dict['gym_backbone'])

def is_timemixer(compoents_dict):
    return (compoents_dict['gym_model'] == 'MLP') and \
           (compoents_dict['gym_series_sampling'] == 'True') and \
           (compoents_dict['gym_series_decomp'] != "None")


class Meta():
    def __init__(self,
                 seed: int=42,
                 task_name: str='LTF',
                 early_stopping=True,
                 batch_size=128,
                 d_model=64,
                 n_layers=2,  # meta-learner层数 (MLP和ICL共用)
                 nhead=4,     # ICL的attention头数
                 dropout=0.1, # dropout比例
                 weight_decay=0.001,
                 lr=0.001,
                 epochs=100,
                 es_tol=5,  # 早停容忍度
                 read_results_root: str='./results_long_term_forecasting',
                 write_results_root: str='./meta/results_long_term_forecasting',
                 ensemble_enabled: bool=False,
                 meta_run_script: str='run_meta_forecast.py',
                 cuda_devices: str='0,1,2,3,4,5,6,7',
                 parallel_workers: int=0,
                 meta_model_type: str='mlp',
                 max_size: int=None,
                 expand_testset: bool=False,
                 meta_script_root: str='./meta/script',
                 rank_aggregation_method: str='median',
                 icl_shuffle: bool=False,
                 icl_batch: bool=False,
                 save_attention: bool=False,
                 k: float=0.0,
                 temporal: float=1.0,
                 suffix: str=''):  # rank聚合方式: mean/median/trimmed_mean/voting; save_attention: 保存ICL attention weights
        self.seed = seed
        self.task_name = task_name
        self.utils = Utils()
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.meta_model_type = meta_model_type
        self.max_size = max_size
        self.expand_testset = expand_testset
        self.meta_script_root = meta_script_root.rstrip('/')
        self.rank_aggregation_method = rank_aggregation_method
        self.icl_shuffle = icl_shuffle
        self.icl_batch = icl_batch
        self.save_attention = save_attention
        # ICL-specific hyperparameters
        self.k = k
        self.temporal = temporal
        self.suffix = suffix

        self.d_model = d_model
        self.n_layers = n_layers
        self.nhead = nhead
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.es_tol = es_tol
        # read from original large-scale results, write new reruns to isolated directory
        self.read_results_root = read_results_root.rstrip('/')
        self.write_results_root = write_results_root.rstrip('/')
        self.ensemble_enabled = ensemble_enabled
        self.meta_run_script = meta_run_script
        # parallel + GPU assignment
        self.cuda_devices = [d.strip() for d in str(cuda_devices).split(',') if d.strip() != '']
        self.parallel_workers = parallel_workers if parallel_workers and parallel_workers > 0 else len(self.cuda_devices)

        # 保存超参数字典，用于后续保存
        self.hyperparams = {
            'seed': seed,
            'batch_size': batch_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'nhead': nhead,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'lr': lr,
            'epochs': epochs,
            'es_tol': es_tol,
            'meta_model_type': meta_model_type,
            'rank_aggregation_method': rank_aggregation_method,
            'icl_shuffle': icl_shuffle,
            'icl_batch': icl_batch,
            'k': k,
            'temporal': temporal,
            'suffix': suffix,
        }
        # Note: nhead is now included above (was missing before)

    def components_processing(self, task_name, datasets, test_dataset, meta_feature_type,
                              pred_len_1=96, pred_len_2=24,
                              metric='mse',
                              components_path='./meta/components.yaml',
                              components_add_Transformer_path='./meta/components_add_Transformer.yaml',
                              components_add_GRU_path='./meta/components_add_GRU.yaml',
                              components_add_LLM_path='./meta/components_add_LLM.yaml',
                              components_add_TSFM_path='./meta/components_add_TSFM.yaml',
                              result_path_MLP=None,
                              result_path_GRU=None,
                              result_path_transformer=None,
                              result_path_LLM=None,
                              result_path_TSFM=None,
                              meta_feature_path='./meta/meta_features',
                              arg_component_balance=False,
                              arg_add_new_dataset=False,
                              arg_add_transformer=False,
                              arg_add_LLM=False,
                              arg_add_TSFM=False,
                              arg_add_GRU=False,
                              arg_all_periods=False,
                              clip_timestamps=False):
        """
        处理组件数据，为K折训练准备数据。
        主要变化：按数据集存储数据，便于后续K折划分。
        """
        self.pred_len_1, self.pred_len_2 = pred_len_1, pred_len_2
        self.clip_timestamps = clip_timestamps
        
        
        self.arg_component_balance = arg_component_balance
        self.arg_add_new_dataset = arg_add_new_dataset
        self.arg_add_transformer = arg_add_transformer
        self.arg_add_LLM = arg_add_LLM
        self.arg_add_TSFM = arg_add_TSFM
        self.arg_add_GRU = arg_add_GRU
        self.arg_all_periods = arg_all_periods
        self.datasets = datasets
        self.test_dataset = test_dataset
        self.meta_feature_type = meta_feature_type
        
        # read paths come from original large-scale experiments
        base_root_read = self.read_results_root
        result_path_MLP = result_path_MLP or os.path.join(base_root_read, 'resultsGym_MLP')
        result_path_GRU = result_path_GRU or os.path.join(base_root_read, 'resultsGym_GRU')
        result_path_transformer = result_path_transformer or os.path.join(base_root_read, 'resultsGym_transformer')
        result_path_LLM = result_path_LLM or os.path.join(base_root_read, 'resultsGym_LLM')
        result_path_TSFM = result_path_TSFM or os.path.join(base_root_read, 'resultsGym_TSFM')
        
        # Filter files by modification time (before Jan 9th 2026)
        import datetime
        self.cutoff_time = datetime.datetime(2026, 1, 18, 0, 0, 0).timestamp()
        logging.info(f'cutoff_time: {self.cutoff_time}')
        file_dict = {}
        file_dict_GRU = {}   # Separate dict for GRU (no completeness filtering)
        file_dict_Transformer = {}  # Separate dict for Transformer (no completeness filtering)
        
        for dataset in datasets:
            # MLP files
            d_path = os.path.join(result_path_MLP, dataset)
            if os.path.exists(d_path):
                if clip_timestamps and dataset != test_dataset:
                    file_dict[dataset] = []
                    for f in os.listdir(d_path):
                        if os.path.exists(os.path.join(d_path, f, 'metrics.npy')):
                            if os.path.getmtime(os.path.join(d_path, f, 'metrics.npy')) < self.cutoff_time:
                                file_dict[dataset].append(f)
                else:
                    file_dict[dataset] = [_ for _ in os.listdir(d_path) if os.path.exists(os.path.join(d_path, _, 'metrics.npy'))]
                
            else:
                file_dict[dataset] = []
            if 'DLinear' in self.suffix:
                filtered_files = []
                for f in file_dict[dataset]:
                    try:
                        if is_DLinear(parse_path(os.path.join(d_path, f))):
                            filtered_files.append(f)
                    except:
                        pass
                logging.info(f"{dataset}: Filtered by DLinear {len(file_dict[dataset])} -> {len(filtered_files)}")
                file_dict[dataset] = filtered_files
            if 'PatchTST' in self.suffix:
                filtered_files = []
                for f in file_dict[dataset]:
                    try:
                        if is_PatchTST(parse_path(os.path.join(d_path, f))):
                            filtered_files.append(f)
                    except:
                        pass
                logging.info(f"{dataset}: Filtered by PatchTST {len(file_dict[dataset])} -> {len(filtered_files)}")
                file_dict[dataset] = filtered_files
            if 'OLinear' in self.suffix:
                filtered_files = []
                for f in file_dict[dataset]:
                    try:
                        if is_OLinear(parse_path(os.path.join(d_path, f))):
                            filtered_files.append(f)
                    except:
                        pass
                logging.info(f"{dataset}: Filtered by OLinear {len(file_dict[dataset])} -> {len(filtered_files)}")
                file_dict[dataset] = filtered_files

            logging.info(f"{dataset}: {len(file_dict[dataset])} MLP files before cutoff time.")
            
            # GRU files (load all, no completeness filtering later)
            if arg_add_GRU:
                d_path_gru = os.path.join(result_path_GRU, dataset)
                if os.path.exists(d_path_gru):
                    file_dict_GRU[dataset] = [_ for _ in os.listdir(d_path_gru) if os.path.exists(os.path.join(d_path_gru, _, 'metrics.npy'))]
                else:
                    file_dict_GRU[dataset] = []
                logging.info(f"{dataset}: {len(file_dict_GRU[dataset])} GRU files loaded (no completeness filter).")
            
            # Transformer files (load all, no completeness filtering later)
            if arg_add_transformer:
                d_path_trans = os.path.join(result_path_transformer, dataset)
                if os.path.exists(d_path_trans):
                    file_dict_Transformer[dataset] = [_ for _ in os.listdir(d_path_trans) if os.path.exists(os.path.join(d_path_trans, _, 'metrics.npy'))]
                else:
                    file_dict_Transformer[dataset] = []
                logging.info(f"{dataset}: {len(file_dict_Transformer[dataset])} Transformer files loaded (no completeness filter).")
        
        # =================== 基于组合完整性的筛选 ===================
        # 只保留四个预测长度都完成的组合，并按数据集限制数量
        logger.info("Filtering combinations by completeness (all 4 pred_lens required)...")
        
        # 定义每个数据集的最大组合数
        max_combos_per_dataset = {'traffic': 250, 'ECL': 250}  # Traffic和ECL取250个
        default_max_combos = 500  # 其他数据集取500个
        
        file_dict_filtered = {}
        for dataset in file_dict.keys():
            files = file_dict[dataset]
            
            # 按 TSGym ID 和 pred_len 分组
            tsgym_predlens = {}  # {tsgym_id: set(pred_lens)}
            tsgym_files = {}     # {tsgym_id: [files]}
            
            for f in files:
                # 提取 TSGym ID
                id_match = re.search(r'(TSGym\d+)', f)
                # 提取 pred_len
                pl_match = re.search(r'_pl(\d+)_', f)
                
                if id_match and pl_match:
                    tsgym_id = id_match.group(1)
                    pl = int(pl_match.group(1))
                    
                    if tsgym_id not in tsgym_predlens:
                        tsgym_predlens[tsgym_id] = set()
                        tsgym_files[tsgym_id] = []
                    tsgym_predlens[tsgym_id].add(pl)
                    tsgym_files[tsgym_id].append(f)
            
            # 根据数据集确定需要的四个 pred_len
            if dataset in ['ili', 'nyse', 'nasdaq']:
                required_pls = {24, 36, 48, 60}
            else:
                required_pls = {96, 192, 336, 720}
            
            # 筛选出四个 pred_len 都有的 TSGym ID
            complete_tsgym_ids = [tid for tid, pls in tsgym_predlens.items()
                                  if required_pls.issubset(pls)]
            
            # 按 TSGym ID 数字排序
            def extract_tsgym_num(tid):
                match = re.search(r'TSGym(\d+)', tid)
                return int(match.group(1)) if match else float('inf')
            
            complete_tsgym_ids_sorted = sorted(complete_tsgym_ids, key=extract_tsgym_num)
            
            # 确定该数据集的最大数量
            max_combos = max_combos_per_dataset.get(dataset, default_max_combos)
            
            # 取前N个完整的 TSGym ID
            selected_tsgym_ids = set(complete_tsgym_ids_sorted[:max_combos])
            
            # 只保留这些 TSGym ID 对应的文件
            filtered_files = []
            for tid in selected_tsgym_ids:
                if tid in tsgym_files:
                    filtered_files.extend(tsgym_files[tid])
            
            file_dict_filtered[dataset] = filtered_files
            
            logger.info(f"  {dataset}: {len(tsgym_predlens)} unique TSGym IDs -> "
                       f"{len(complete_tsgym_ids)} complete (4 pls) -> "
                       f"{len(selected_tsgym_ids)} selected (max {max_combos}) -> "
                       f"{len(filtered_files)} files")
        
        file_dict = file_dict_filtered
        logger.info(f"After completeness filtering: {sum([len(_) for _ in file_dict.values()])} total files")
        
        if arg_all_periods:
            # 当 arg_all_periods=True 时：训练集保留所有 pred_len，测试集只用当前 pred_len
            file_dict_test = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k in ['ili','nyse','nasdaq'] 
                            else [_ for _ in v if f'pl{pred_len_1}' in _] for k, v in file_dict.items()}
        else:
            file_dict = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k in ['ili','nyse','nasdaq'] 
                       else [_ for _ in v if f'pl{pred_len_1}' in _]for k, v in file_dict.items()}
            file_dict_test = file_dict.copy()

        logger.info(f'number of combinations: {sum([len(_) for _ in file_dict.values()])}')

        # Limit file_dict by max_size if specified
        if self.max_size is not None:
            logger.info(f'Limiting results pool to max_size={self.max_size} by TSGym ID')
            file_dict_limited = {}
            for dataset in file_dict.keys():
                files = file_dict[dataset]
                def extract_tsgym_id(filename):
                    match = re.search(r'TSGym(\d+)', filename)
                    if match:
                        return int(match.group(1))
                    return float('inf')
                
                files_sorted = sorted(files, key=extract_tsgym_id)
                files_limited = files_sorted[:self.max_size]
                file_dict_limited[dataset] = files_limited
                logger.info(f'{dataset}: {len(files)} -> {len(files_limited)} files after max_size limit')
            file_dict = file_dict_limited
            logger.info(f'After max_size limit: {sum([len(_) for _ in file_dict.values()])} combinations')
            if arg_all_periods:
                # 当 arg_all_periods=True 时，测试集只用当前 pred_len
                file_dict_test = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k in ['ili','nyse','nasdaq'] 
                                else [_ for _ in v if f'pl{pred_len_1}' in _] for k, v in file_dict.items()}
            else:
                file_dict_test = file_dict.copy()

        if arg_component_balance:
            logger.info(f'Before balance: {sum([len(_) for _ in file_dict.values()])}')
            file_dict_resample = file_dict.copy()
            num_intersection = min([len(_) for _ in file_dict_resample.values()])
            file_dict_resample = {k: random.sample(v, num_intersection) for k,v in file_dict_resample.items()}
            file_dict = file_dict_resample
            logger.info(f'After balance: {sum([len(_) for _ in file_dict.values()])}')

        # =================== Merge GRU/Transformer files (no completeness filtering) ===================
        # These pools have incomplete experiments, so we only filter by pred_len
        if arg_add_GRU and file_dict_GRU:
            logger.info("Merging GRU files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_GRU.keys():
                gru_files = file_dict_GRU[dataset]
                # Filter by pred_len
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    gru_files_filtered = [f for f in gru_files if f'pl{pred_len_2}' in f]
                else:
                    gru_files_filtered = [f for f in gru_files if f'pl{pred_len_1}' in f]
                
                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(gru_files_filtered)
                logger.info(f"  {dataset}: Added {len(gru_files_filtered)} GRU files")
        
        if arg_add_transformer and file_dict_Transformer:
            logger.info("Merging Transformer files (filtered by pred_len only, no completeness check)...")
            for dataset in file_dict_Transformer.keys():
                trans_files = file_dict_Transformer[dataset]
                # Filter by pred_len
                if dataset in ['ili', 'nyse', 'nasdaq']:
                    trans_files_filtered = [f for f in trans_files if f'pl{pred_len_2}' in f]
                else:
                    trans_files_filtered = [f for f in trans_files if f'pl{pred_len_1}' in f]
                
                if dataset not in file_dict:
                    file_dict[dataset] = []
                file_dict[dataset].extend(trans_files_filtered)
                logger.info(f"  {dataset}: Added {len(trans_files_filtered)} Transformer files")
        
        logger.info(f'After merging GRU/Transformer: {sum([len(_) for _ in file_dict.values()])} total files')
        
        # Also update file_dict_test if GRU/Transformer is added
        if arg_add_GRU or arg_add_transformer:
            if arg_all_periods:
                file_dict_test = {k: [_ for _ in v if f'pl{pred_len_2}' in _] if k in ['ili','nyse','nasdaq'] 
                                else [_ for _ in v if f'pl{pred_len_1}' in _] for k, v in file_dict.items()}
            else:
                file_dict_test = file_dict.copy()

        # load meta features
        meta_feature_path = os.path.join(meta_feature_path, f"meta_feature_dict_{meta_feature_type}.npz")
        meta_features = np.load(meta_feature_path, allow_pickle=True)
        self.name_dict = {dataset: dataset for dataset in datasets}
        self.name_dict['ECL'] = 'electricity'
        self.name_dict['Exchange'] = 'exchange_rate'
        self.name_dict['ili'] = 'national_illness'
        self.meta_features = {key: meta_features[self.name_dict[key]] for key in datasets}
        self.meta_features = {k: v.flatten() for k,v in self.meta_features.items()}
        
        assert len(set([v.shape for v in self.meta_features.values()])) == 1
        self.meta_feature_dim = list(self.meta_features.values())[0].shape[0]
        
        # z-score on different datasets
        mu = np.nanmean(np.stack(list(self.meta_features.values())), axis=0)
        std = np.nanstd(np.stack(list(self.meta_features.values())), axis=0)
        self.meta_features = {k: (v - mu) / (std + 1e-6) for k,v in self.meta_features.items()}
        self.meta_features = {k: np.clip(v, -1e4, 1e4) for k, v in self.meta_features.items()}
        self.meta_features = {k: np.where(np.isnan(v), 0, v) for k, v in self.meta_features.items()}
        assert (~np.isnan(np.stack(list(self.meta_features.values())))).all()
        
        # training datasets and testing dataset
        datasets_train = [_ for _ in datasets if _ != test_dataset]
        if not arg_add_new_dataset:
            datasets_train = [_ for _ in datasets_train if _ not in ['covid-19', 'fred-md']]
        self.datasets_train = datasets_train
        dataset_test = [_ for _ in datasets if _ == test_dataset][0]
        logger.info(f'training datasets: {datasets_train}, testing dataset: {dataset_test}')
        
        # load components - start with base, then merge additional components
        with open(components_path, 'r') as f:
            self.components = yaml.safe_load(f)
        
        # Merge Transformer components (extend lists for each key)
        if arg_add_transformer:
            with open(components_add_Transformer_path, 'r') as f:
                trans_components = yaml.safe_load(f)
            for k, v in trans_components.items():
                if k in self.components:
                    # Extend existing list with new values (avoiding duplicates)
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged Transformer components from {components_add_Transformer_path}")
        
        # Merge GRU components
        if arg_add_GRU:
            with open(components_add_GRU_path, 'r') as f:
                gru_components = yaml.safe_load(f)
            for k, v in gru_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged GRU components from {components_add_GRU_path}")
        
        # Merge LLM components
        if arg_add_LLM:
            with open(components_add_LLM_path, 'r') as f:
                llm_components = yaml.safe_load(f)
            for k, v in llm_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged LLM components from {components_add_LLM_path}")
        
        # Merge TSFM components    
        if arg_add_TSFM:
            with open(components_add_TSFM_path, 'r') as f:
                tsfm_components = yaml.safe_load(f)
            for k, v in tsfm_components.items():
                if k in self.components:
                    existing = set(self.components[k])
                    self.components[k] = self.components[k] + [x for x in v if x not in existing]
                else:
                    self.components[k] = v
            logger.info(f"Merged TSFM components from {components_add_TSFM_path}")

        if self.arg_all_periods:
            self.components['gym_pl'] = ['24', '36', '48', '60'] + ['96', '192', '336', '720']
        
        # 创建并保存LabelEncoders
        self.label_encoders = {}
        components_encoded = {}
        for k, v in self.components.items():
            le = preprocessing.LabelEncoder()
            le.fit(v)
            self.label_encoders[k] = le
            components_encoded[k] = {kk: vv for kk, vv in zip(v, le.transform(v))}
        self.components = components_encoded

        # =================== 按数据集存储数据 ===================
        # 为K折训练准备：按数据集分别存储组件、meta-features和targets
        self.dataset_data = {}  # {dataset_name: {'components': [], 'meta_features': [], 'targets': [], 'names': []}}
        
        for dataset in file_dict.keys():
            dataset_components = []
            dataset_meta_features = []
            dataset_targets = []
            dataset_names = []
            
            for _ in file_dict[dataset]:
                if 'Transformer' in _:
                    result_path = result_path_transformer
                elif 'LLM' in _:
                    result_path = result_path_LLM
                elif 'TSFM' in _:
                    result_path = result_path_TSFM
                elif 'MLP' in _:
                    result_path = result_path_MLP
                elif 'GRU' in _:
                    result_path = result_path_GRU
                else:
                    raise NotImplementedError
                
                idx = 0 if metric == 'mae' else 1
                try:
                    metric_value = np.load(f'{result_path}/{dataset}/{_}/metrics.npy')[idx]
                except FileNotFoundError:
                    continue
                
                # 跳过nan值的样本（实验可能失败或未完成）
                if np.isnan(metric_value):
                    continue
                
                k = '_'.join([dataset, _])
                k2 = k.replace(f'{task_name}_', '')
                k_HP = '_'.join(k2[re.search(r'TSGym\d*', k2).end()+1: ].split('_')[12:])
                current_components = k2[re.search(r'TSGym\d*', k2).end()+1: ].split('_')[:12] +\
                                      [re.search(r'_sl(\d+)_', k_HP).group(1),
                                       re.search(r'_dm(\d+)_', k_HP).group(1),
                                       re.search(r'_df(\d+)_', k_HP).group(1),
                                       re.search(r'_el(\d+)_', k_HP).group(1),
                                       re.search(r'_epochs(\d+)_', k_HP).group(1),
                                       re.search(r'lf([^_]+)', k_HP).group(1),
                                       re.search(r'_lr([\d.]+)_', k_HP).group(1),
                                       re.search(r'lrs([^_]+)', k_HP).group(1)]
                if self.arg_all_periods: 
                    current_components = current_components + [re.search(r'_pl(\d+)_', k_HP).group(1)]
                
                assert len(current_components) == len(self.components)
                current_components_dict = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
                try:
                    current_components_encoded = [self.components[k][v] for k, v in current_components_dict.items()]
                except KeyError:
                    logger.info(f'{k} is not in components')
                    continue
                
                dataset_components.append(current_components_encoded)
                dataset_meta_features.append(self.meta_features[dataset])
                dataset_targets.append(metric_value)
                dataset_names.append(k)
            
            if len(dataset_components) > 0:
                self.dataset_data[dataset] = {
                    'components': np.array(dataset_components),
                    'meta_features': np.array(dataset_meta_features),
                    'targets': np.array(dataset_targets),
                    'names': dataset_names
                }
                logger.info(f"Dataset {dataset}: {len(dataset_components)} samples loaded")

        # =================== Load Few-Shot Data for 'fewshot-mlp' ===================
        if "real-fewshot" in self.suffix:
            # Real few-shot: use top 50 TSGym IDs from the current test dataset
            real_fewshot_dataset_name = f"real_fewshot_{test_dataset}"
            logger.info(f"Generating real few-shot data for {test_dataset} from top 50 TSGym IDs...")
            
            if test_dataset in self.dataset_data:
                data = self.dataset_data[test_dataset]
                names = data['names']
                
                # Extract TSGym IDs
                tsgym_ids = []
                valid_indices = []
                for idx, name in enumerate(names):
                    # name format is usually dataset_TSGym123_...
                    # we search for TSGym followed by digits
                    match = re.search(r'TSGym(\d+)', name)
                    if match:
                        tsgym_ids.append(int(match.group(1)))
                        valid_indices.append(idx)
                
                if len(tsgym_ids) > 0:
                    unique_ids = sorted(list(set(tsgym_ids)))
                    # Take top 50 IDs
                    top_50_ids = set(unique_ids[:500])
                    logger.info(f"Selected {len(top_50_ids)} unique TSGym IDs for few-shot (Range: {min(top_50_ids)} - {max(top_50_ids)})")
                    
                    # Filter samples
                    fs_indices = [valid_indices[i] for i, tid in enumerate(tsgym_ids) if tid in top_50_ids]
                    
                    if len(fs_indices) > 0:
                        fs_components = data['components'][fs_indices] # shape (N, D)
                        fs_meta = data['meta_features'][fs_indices]   # shape (N, M)
                        fs_targets = data['targets'][fs_indices]      # shape (N,)
                        fs_names = [data['names'][i] for i in fs_indices]
                        
                        # Resample to target size (500 or 250)
                        # Traffic and ECL use 250, others 500
                        target_size = 250 if test_dataset in ['traffic', 'ECL'] else 500
                        current_len = len(fs_indices)
                        
                        if current_len < target_size:
                            repeat_factor = int(np.ceil(target_size / current_len))
                            
                            # Tile using numpy logic
                            fs_components = np.tile(fs_components, (repeat_factor, 1))[:target_size]
                            fs_meta = np.tile(fs_meta, (repeat_factor, 1))[:target_size]
                            fs_targets = np.tile(fs_targets, (repeat_factor))[:target_size]
                            
                            # Tile list of names
                            fs_names = (fs_names * repeat_factor)[:target_size]
                            
                            logger.info(f"Resampled real few-shot from {current_len} to {target_size}")
                        
                        # Store in dataset_data
                        self.dataset_data[real_fewshot_dataset_name] = {
                            'components': fs_components,
                            'meta_features': fs_meta,
                            'targets': fs_targets,
                            'names': fs_names
                        }
                        
                        # Add to training datasets
                        if real_fewshot_dataset_name not in self.datasets_train:
                            self.datasets_train.append(real_fewshot_dataset_name)
                            logger.info(f"Added {real_fewshot_dataset_name} to training datasets.")
                            
                    else:
                        logger.warning(f"No samples found corresponding to top 50 IDs in {test_dataset}")
                else:
                    logger.warning(f"No valid TSGym IDs parsed in {test_dataset}")
            else:
                logger.warning(f"Test dataset {test_dataset} not found in dataset_data. Cannot generate real few-shot.")

        elif "all-fewshot" in self.suffix:
            for target_dataset in self.datasets:
                fewshot_dataset_name = f"fewshot_{target_dataset}"
                logger.info(f"Loading few-shot results for {target_dataset} as training data ({fewshot_dataset_name})...")
                
                result_path_fewshot = './results_long_term_forecasting_fewshot/resultsGym_MLP'
                d_path_fewshot = os.path.join(result_path_fewshot, target_dataset)
                
                if os.path.exists(d_path_fewshot):
                    fs_files = [_ for _ in os.listdir(d_path_fewshot) if os.path.exists(os.path.join(d_path_fewshot, _, 'metrics.npy'))]
                    
                    fs_components = []
                    fs_meta_features = []
                    fs_targets = []
                    fs_names = []
                    
                    for f in fs_files:
                        target_pl_train = pred_len_2 if 'ili' in target_dataset or 'nyse' in target_dataset or 'nasdaq' in target_dataset else pred_len_1
                        if f'pl{target_pl_train}' not in f:
                            continue
                        idx = 0 if metric == 'mae' else 1
                        try:
                            metric_val = np.load(os.path.join(d_path_fewshot, f, 'metrics.npy'))[idx]
                        except:
                            continue
                        
                        if np.isnan(metric_val): continue

                        k = '_'.join([target_dataset, f])
                        k2 = k.replace(f'{task_name}_', '')
                        # Extract TSGym ID end index
                        match_tsgym = re.search(r'TSGym\d*', k2)
                        if not match_tsgym: continue
                        
                        try:
                            suffix_part = k2[match_tsgym.end()+1: ]
                            parts = suffix_part.split('_')
                            if len(parts) < 12: continue
                            
                            base_comps = parts[:12]
                            hp_part_str = '_'.join(parts[12:])
                            
                            # Parsing HPs using regex from main loop logic
                            sl = re.search(r'_sl(\d+)_', hp_part_str).group(1)
                            dm = re.search(r'_dm(\d+)_', hp_part_str).group(1)
                            df = re.search(r'_df(\d+)_', hp_part_str).group(1)
                            el = re.search(r'_el(\d+)_', hp_part_str).group(1)
                            epochs = re.search(r'_epochs(\d+)_', hp_part_str).group(1)
                            lf = re.search(r'lf([^_]+)', hp_part_str).group(1)
                            lr = re.search(r'_lr([\d.]+)_', hp_part_str).group(1)
                            lrs = re.search(r'lrs([^_]+)', hp_part_str).group(1)
                            
                            current_components = base_comps + [sl, dm, df, el, epochs, lf, lr, lrs]
                            
                            # Encode
                            current_components_dict = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
                            current_components_encoded = [self.components[k][v] for k, v in current_components_dict.items()]
                            
                            fs_components.append(current_components_encoded)
                            # Use meta features of target_dataset
                            fs_meta_features.append(self.meta_features[target_dataset])
                            fs_targets.append(metric_val)
                            fs_names.append(k)
                            
                        except Exception as e:
                            # logger.warning(f"Failed to parse fewshot file {f}: {e}")
                            continue

                    if len(fs_components) > 0:
                        self.dataset_data[fewshot_dataset_name] = {
                            'components': np.array(fs_components),
                            'meta_features': np.array(fs_meta_features),
                            'targets': np.array(fs_targets),
                            'names': fs_names
                        }
                        logger.info(f"Loaded {len(fs_components)} few-shot samples for {fewshot_dataset_name}")
                    else:
                        logger.info(f"No few-shot samples found for {fewshot_dataset_name}")
                else:
                    logger.warning(f"Few-shot directory not found: {d_path_fewshot}")

        elif "fewshot" in self.suffix:
            fewshot_dataset_name = f"fewshot_{test_dataset}"
            logger.info(f"Loading few-shot results for {test_dataset} as training data ({fewshot_dataset_name})...")
            
            result_path_fewshot = './results_long_term_forecasting_fewshot/resultsGym_MLP'
            d_path_fewshot = os.path.join(result_path_fewshot, test_dataset)
            
            if os.path.exists(d_path_fewshot):
                fs_files = [_ for _ in os.listdir(d_path_fewshot) if os.path.exists(os.path.join(d_path_fewshot, _, 'metrics.npy'))]
                
                fs_components = []
                fs_meta_features = []
                fs_targets = []
                fs_names = []
                
                for f in fs_files:
                    target_pl_train = pred_len_2 if 'ili' in test_dataset or 'nyse' in test_dataset or 'nasdaq' in test_dataset else pred_len_1
                    if f'pl{target_pl_train}' not in f:
                        continue
                    idx = 0 if metric == 'mae' else 1
                    try:
                        metric_val = np.load(os.path.join(d_path_fewshot, f, 'metrics.npy'))[idx]
                    except:
                        continue
                    
                    if np.isnan(metric_val): continue

                    k = '_'.join([test_dataset, f])
                    k2 = k.replace(f'{task_name}_', '')
                    # Extract TSGym ID end index
                    match_tsgym = re.search(r'TSGym\d*', k2)
                    if not match_tsgym: continue
                    
                    try:
                        suffix_part = k2[match_tsgym.end()+1: ]
                        parts = suffix_part.split('_')
                        if len(parts) < 12: continue
                        
                        base_comps = parts[:12]
                        hp_part_str = '_'.join(parts[12:])
                        
                        # Parsing HPs using regex from main loop logic
                        sl = re.search(r'_sl(\d+)_', hp_part_str).group(1)
                        dm = re.search(r'_dm(\d+)_', hp_part_str).group(1)
                        df = re.search(r'_df(\d+)_', hp_part_str).group(1)
                        el = re.search(r'_el(\d+)_', hp_part_str).group(1)
                        epochs = re.search(r'_epochs(\d+)_', hp_part_str).group(1)
                        lf = re.search(r'lf([^_]+)', hp_part_str).group(1)
                        lr = re.search(r'_lr([\d.]+)_', hp_part_str).group(1)
                        lrs = re.search(r'lrs([^_]+)', hp_part_str).group(1)
                        
                        current_components = base_comps + [sl, dm, df, el, epochs, lf, lr, lrs]
                        
                        # Encode
                        current_components_dict = {list(self.components.keys())[i]: v for i, v in enumerate(current_components)}
                        current_components_encoded = [self.components[k][v] for k, v in current_components_dict.items()]
                        
                        fs_components.append(current_components_encoded)
                        # Use meta features of test_dataset
                        fs_meta_features.append(self.meta_features[test_dataset])
                        fs_targets.append(metric_val)
                        fs_names.append(k)
                        
                    except Exception as e:
                        # logger.warning(f"Failed to parse fewshot file {f}: {e}")
                        continue

                if len(fs_components) > 0:
                    self.dataset_data[fewshot_dataset_name] = {
                        'components': np.array(fs_components),
                        'meta_features': np.array(fs_meta_features),
                        'targets': np.array(fs_targets),
                        'names': fs_names
                    }
                    logger.info(f"Loaded {len(fs_components)} few-shot samples for {fewshot_dataset_name}")
                else:
                    logger.info(f"No few-shot samples found for {fewshot_dataset_name}")
            else:
                logger.warning(f"Few-shot directory not found: {d_path_fewshot}")

        # =================== 处理测试集 ===================
        self.test_script_map = {}
        testset_components, testset_meta_features, testset_targets, testset_targets_mae = [], [], [], []
        self.name_components = []
        
        if self.expand_testset:
            import glob
            target_pl = pred_len_2 if dataset_test in ['ili', 'nyse', 'nasdaq', 'covid-19', 'fred-md'] else pred_len_1
            allowed_gym_types = ['MLP']
            if self.arg_add_GRU:
                allowed_gym_types.append('GRU')
            if self.arg_add_transformer:
                allowed_gym_types.append('Transformer')
            if self.arg_add_LLM:
                allowed_gym_types.append('LLM')
            if self.arg_add_TSFM:
                allowed_gym_types.append('TSFM')
            
            dataset_dir = os.path.join(self.meta_script_root, dataset_test)
            script_paths = []
            for gym_type in allowed_gym_types:
                pattern = os.path.join(dataset_dir, gym_type, 'random', f'pred_len_{target_pl}', '*.sh')
                found_paths = glob.glob(pattern)
                script_paths.extend(found_paths)
                if found_paths:
                    logger.info(f"[expand_testset] {gym_type}: found {len(found_paths)} scripts")
            
            script_paths = sorted(script_paths)
            logger.info(f"[expand_testset] total scripts loaded: {len(script_paths)}")

        def _parse_components_from_script_model_name(model_name: str):
            parts = model_name.split('_')
            try:
                hp_idx = parts.index('HP')
            except ValueError:
                return None
            base_parts = parts[:hp_idx]
            hp_parts = parts[hp_idx+1:]
            if len(base_parts) < 13:
                return None
            comps12 = base_parts[1:13]
            if len(hp_parts) < 7:
                return None
            seqlen = hp_parts[0]
            dm_df = hp_parts[1]
            if '-' not in dm_df:
                return None
            dm, df = dm_df.split('-', 1)
            el = hp_parts[2]
            epochs = hp_parts[3]
            loss = hp_parts[4]
            lr = hp_parts[5]
            lrs = hp_parts[6]
            comps = comps12 + [seqlen, dm, df, el, epochs, loss, lr, lrs]
            if self.arg_all_periods:
                pl_match = re.search(r'_pl(\d+)$', model_name)
                if pl_match:
                    comps.append(pl_match.group(1))
            return comps

        if self.expand_testset:
            # 遍历脚本并解析组件
            for sp in script_paths:
                fn = os.path.basename(sp)
                suffix = f"_pl{target_pl}.sh"
                if not fn.endswith(suffix):
                    continue
                model_name = fn[:-len(suffix)]
                self.test_script_map[model_name] = sp
                
                current_components = _parse_components_from_script_model_name(model_name)
                if current_components is None:
                    continue
                assert len(current_components) == len(self.components)
                current_components_dict = {list(self.components.keys())[i]: vv for i, vv in enumerate(current_components)}
                try:
                    current_components_encoded = [self.components[kk][vv] for kk, vv in current_components_dict.items()]
                except KeyError:
                    continue
                
                testset_components.append(current_components_encoded)
                testset_meta_features.append(self.meta_features[dataset_test])
                testset_targets.append(np.nan)
                testset_targets_mae.append(np.nan)
                self.name_components.append(model_name)
        else:
            # 从已有结果加载测试集
            for _ in file_dict_test.get(dataset_test, []):
                if 'Transformer' in _:
                    result_path = result_path_transformer
                elif 'LLM' in _:
                    result_path = result_path_LLM
                elif 'TSFM' in _:
                    result_path = result_path_TSFM
                elif 'MLP' in _:
                    result_path = result_path_MLP
                elif 'GRU' in _:
                    result_path = result_path_GRU
                else:
                    raise NotImplementedError
                
                try:
                    metrics = np.load(f'{result_path}/{dataset_test}/{_}/metrics.npy')
                    metric_mse = metrics[1]
                    metric_mae = metrics[0]
                except FileNotFoundError:
                    continue
                
                # 跳过nan值的样本（实验可能失败或未完成）
                if np.isnan(metric_mse) or np.isnan(metric_mae):
                    continue
                
                k = '_'.join([dataset_test, _])
                k2 = k.replace(f'{task_name}_', '')
                k_HP = '_'.join(k2[re.search(r'TSGym\d*', k2).end()+1: ].split('_')[12:])
                current_components = k2[re.search(r'TSGym\d*', k2).end()+1: ].split('_')[:12] +\
                                      [re.search(r'_sl(\d+)_', k_HP).group(1),
                                       re.search(r'_dm(\d+)_', k_HP).group(1),
                                       re.search(r'_df(\d+)_', k_HP).group(1),
                                       re.search(r'_el(\d+)_', k_HP).group(1),
                                       re.search(r'_epochs(\d+)_', k_HP).group(1),
                                       re.search(r'lf([^_]+)', k_HP).group(1),
                                       re.search(r'_lr([\d.]+)_', k_HP).group(1),
                                       re.search(r'lrs([^_]+)', k_HP).group(1)]
                if self.arg_all_periods: 
                    current_components = current_components + [re.search(r'_pl(\d+)_', k_HP).group(1)]
                assert len(current_components) == len(self.components)
                current_components_dict = {list(self.components.keys())[i]: vv for i, vv in enumerate(current_components)}
                try:
                    current_components_encoded = [self.components[kk][vv] for kk, vv in current_components_dict.items()]
                except KeyError:
                    continue
                
                testset_components.append(current_components_encoded)
                testset_meta_features.append(self.meta_features[dataset_test])
                testset_targets.append(metric_mse)
                testset_targets_mae.append(metric_mae)
                self.name_components.append(k)

        # 转换为tensor
        self.device = self.utils.get_device()
        
        if len(testset_components) > 0:
            self.testset_components = torch.from_numpy(np.stack(testset_components)).long().to(self.device)
            self.testset_meta_features = torch.from_numpy(np.stack(testset_meta_features)).float().to(self.device)
            self.testset_targets = torch.from_numpy(np.stack(testset_targets)).float().to(self.device)
            self.testset_targets_mae = torch.from_numpy(np.stack(testset_targets_mae)).float().to(self.device)
        else:
            raise ValueError(f"No test samples found for {dataset_test}")
        
        logger.info(f"Test set: {len(testset_components)} samples")
        
        # =================== Save TSGymIDs per dataset for reproducibility ===================
        self.dataset_tsgym_ids = {}  # {dataset_name: [list of unique TSGymIDs]}
        for dataset, data in self.dataset_data.items():
            tsgym_ids = set()
            for name in data['names']:
                match = re.search(r'(TSGym\d+)', name)
                if match:
                    tsgym_ids.add(match.group(1))
            self.dataset_tsgym_ids[dataset] = sorted(list(tsgym_ids), key=lambda x: int(re.search(r'\d+', x).group()))
            logger.info(f"  {dataset}: {len(self.dataset_tsgym_ids[dataset])} unique TSGymIDs saved")
        
        # Save to file for reproducibility
        corpus_info_path = (
            f'./meta/results/corpus_tsgym_ids_{self.meta_feature_type}'
            f'_test_{self.test_dataset}'
            f'_pl{self.pred_len_1}_{self.pred_len_2}'
            f'_gru_{self.arg_add_GRU}'
            f'_trans_{self.arg_add_transformer}.json'
        )
        os.makedirs(os.path.dirname(corpus_info_path), exist_ok=True)
        import json
        corpus_info = {
            'test_dataset': self.test_dataset,
            'pred_len_1': self.pred_len_1,
            'pred_len_2': self.pred_len_2,
            'arg_add_GRU': self.arg_add_GRU,
            'arg_add_transformer': self.arg_add_transformer,
            'dataset_tsgym_ids': self.dataset_tsgym_ids,
            'total_samples_per_dataset': {d: len(data['names']) for d, data in self.dataset_data.items()}
        }
        with open(corpus_info_path, 'w') as f:
            json.dump(corpus_info, f, indent=2)
        logger.info(f"Corpus TSGymIDs saved to: {corpus_info_path}")

    def prepare_fold_data(self, val_dataset):
        """
        为单折训练准备数据。
        
        Args:
            val_dataset: 用作验证集的数据集名称
        
        Returns:
            train_data, val_data: 训练集和验证集的数据字典
        """
        train_components, train_meta_features, train_targets_rank = [], [], []
        val_components, val_meta_features, val_targets_rank = [], [], []
        
        for dataset, data in self.dataset_data.items():
            if dataset == self.test_dataset:
                continue  # 跳过测试数据集
            
            # 对每个数据集内部分别做 rank（归一化到 [1/n, 1]）
            targets = data['targets']
            targets_rank = (np.argsort(np.argsort(targets)).astype(np.float32) + 1) / len(targets)
            
            if dataset == val_dataset:
                # 作为验证集
                val_components.append(data['components'])
                val_meta_features.append(data['meta_features'])
                val_targets_rank.append(targets_rank)
            else:
                # 作为训练集
                train_components.append(data['components'])
                train_meta_features.append(data['meta_features'])
                train_targets_rank.append(targets_rank)
        
        # 合并训练数据（rank 已经在每个数据集内部计算好了）
        train_components = np.concatenate(train_components, axis=0)
        train_meta_features = np.concatenate(train_meta_features, axis=0)
        train_targets_rank = np.concatenate(train_targets_rank, axis=0)
        
        # 合并验证数据
        val_components = np.concatenate(val_components, axis=0)
        val_meta_features = np.concatenate(val_meta_features, axis=0)
        val_targets_rank = np.concatenate(val_targets_rank, axis=0)
        
        # 转换为tensor
        train_data = {
            'components': torch.from_numpy(train_components).long().to(self.device),
            'meta_features': torch.from_numpy(train_meta_features).float().to(self.device),
            'targets': torch.from_numpy(train_targets_rank).float().to(self.device),
        }
        val_data = {
            'components': torch.from_numpy(val_components).long().to(self.device),
            'meta_features': torch.from_numpy(val_meta_features).float().to(self.device),
            'targets': torch.from_numpy(val_targets_rank).float().to(self.device),
        }
        
        return train_data, val_data
        
    def loss_pearson(self, y_pred, y_true, cal_loss=True):
        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        if cal_loss:
            return 1 - corr
        else:
            return corr
        
    def get_cross_dataset_performance(self, best_setting_name):
        cross_performance = {}
        import glob
        
        match = re.search(r'(TSGym\d+)', best_setting_name)
        if not match:
            logger.warning(f"Could not extract TSGym ID from {best_setting_name}")
            cross_performance = {d: np.array([np.nan, np.nan]) for d in self.datasets if d != self.test_dataset}
            return cross_performance
        tsgym_id = match.group(1)
        
        logger.info(f"Finding cross-dataset performance for ID: {tsgym_id} (Best on {self.test_dataset})")

        for dset in self.datasets:
            if dset == self.test_dataset:
                continue
            
            if dset in ['ili', 'nyse', 'nasdaq']: 
                t_pl = self.pred_len_2
            else:
                t_pl = self.pred_len_1
            
            possible_names = {dset}
            if hasattr(self, 'name_dict') and dset in self.name_dict:
                 possible_names.add(self.name_dict[dset])
            
            found_metrics = None
            
            for name_variant in possible_names:
                search_pattern = f"{self.read_results_root}/resultsGym_*/{name_variant}/LTF_{tsgym_id}_*_pl{t_pl}_*"
                res = glob.glob(search_pattern)
                
                if res:
                    res.sort(key=os.path.getmtime, reverse=True)
                    best_folder = res[0]
                    metric_file = os.path.join(best_folder, 'metrics.npy')
                    if os.path.exists(metric_file):
                        try:
                            metrics = np.load(metric_file)
                            found_metrics = metrics[:2]
                            break 
                        except Exception as e:
                            logger.error(f"Error reading metrics from {metric_file}: {e}")
            
            if found_metrics is not None:
                cross_performance[dset] = found_metrics
            else:
                cross_performance[dset] = np.array([np.nan, np.nan])
                
        return cross_performance

    def ensemble_predictions(self, topk_names, dataset_test, root_path=None, run_script=None):
        preds = []
        collected_tsgym_ids = []  # Track TSGym IDs for each collected prediction
        trues = None
        import glob
        import subprocess
        import shutil
        import concurrent.futures

        root_path = (root_path or self.write_results_root).rstrip('/')
        run_script = run_script or self.meta_run_script

        for setting_name in topk_names:
            match = re.search(r'(TSGym\d+)', setting_name)
            if not match:
                logger.warning(f"Could not extract TSGym ID from {setting_name}")
                continue
            tsgym_id = match.group(1)
            
            pl_match = re.search(r'_pl(\d+)_', setting_name)
            target_pl = pl_match.group(1) if pl_match else None

            def find_result_folder(tid, dset, t_pl):
                search_pattern = f"{root_path}/resultsGym_*/{dset}/LTF_{tid}_*_pl{t_pl}_*"
                res = glob.glob(search_pattern)
                
                if not res:
                     search_pattern_ckpt = f"checkpointsGym/LTF_{tid}_*_{dset}*"
                     res = glob.glob(search_pattern_ckpt)

                if res:
                    logger.info("Found result folders:")
                    logger.info(res)
                    res.sort(key=os.path.getmtime, reverse=True)
                    return res[0]
                return None

            res_folder = find_result_folder(tsgym_id, dataset_test, target_pl)
            logger.info(f"Result folder for {tsgym_id}, dataset={dataset_test}, pl={target_pl}: {res_folder}")
            result_ready = False
            true_file_fallback = None  # 用于存储从其他文件夹找到的 true.npy 路径
            
            if res_folder:
                pred_file = os.path.join(res_folder, 'pred.npy')
                true_file = os.path.join(res_folder, 'true.npy')
                if os.path.exists(pred_file) and os.path.exists(true_file):
                    result_ready = True
                elif os.path.exists(pred_file) and not os.path.exists(true_file):
                    # pred.npy 存在但 true.npy 缺失
                    # y_true 只和 dataset 和 pred_len 有关，尝试从其他结果文件夹找 true.npy
                    logger.info(f"pred.npy exists but true.npy missing for {tsgym_id}. Searching for true.npy from other folders...")
                    search_pattern_true = f"{root_path}/resultsGym_*/{dataset_test}/LTF_*_pl{target_pl}_*/true.npy"
                    true_candidates = glob.glob(search_pattern_true)
                    if true_candidates:
                        # 选择最新的或第一个
                        true_candidates.sort(key=os.path.getmtime, reverse=True)
                        true_file_fallback = true_candidates[0]
                        logger.info(f"Found fallback true.npy: {true_file_fallback}")
                        result_ready = True
                    else:
                        logger.warning(f"Could not find any true.npy for dataset={dataset_test}, pl={target_pl}")
            
            if not result_ready:
                logger.info(f"Results missing for ID {tsgym_id} (dataset={dataset_test}). Searching for script...")
                
                script_base_dir = "scripts/long_term_forecast"
                target_dir_name = next((d for d in os.listdir(script_base_dir) if d.lower() == f"{dataset_test}_script".lower()), None)
                
                if target_dir_name:
                    script_pattern = f"{script_base_dir}/{target_dir_name}/**/{tsgym_id}_*.sh"
                    found_scripts = glob.glob(script_pattern, recursive=True)
                else:
                    logger.warning(f"Could not find script directory matching {dataset_test}_script.")
                    found_scripts = []
                
                target_script = None
                if found_scripts and target_pl:
                    for script in found_scripts:
                        try:
                            with open(script, 'r') as f:
                                content = f.read()
                                if f"--pred_len {target_pl}" in content or f"pred_len {target_pl}" in content:
                                    target_script = script
                                    break
                        except Exception as e:
                            logger.error(f"Error reading script {script}: {e}")
                elif found_scripts:
                    target_script = found_scripts[0]
                
                if target_script:
                    if res_folder and os.path.exists(res_folder):
                        shutil.rmtree(res_folder)
                    if 'to_run' not in locals():
                        to_run = []
                    to_run.append((target_script, tsgym_id, target_pl))
                else:
                    logger.warning(f"No suitable script found for {tsgym_id} (dataset={dataset_test}, pl={target_pl})")

            if result_ready:
                try:
                    pred = np.load(pred_file)
                    preds.append(pred)
                    collected_tsgym_ids.append(tsgym_id)  # Record the TSGym ID
                    if trues is None:
                        # 优先使用原始 true_file，如果不存在则使用 fallback
                        actual_true_file = true_file if os.path.exists(true_file) else true_file_fallback
                        if actual_true_file and os.path.exists(actual_true_file):
                            trues = np.load(actual_true_file)
                        else:
                            logger.warning(f"No valid true.npy found for loading (tried: {true_file}, fallback: {true_file_fallback})")
                except Exception as e:
                     logger.info(f"Error loading npy files: {e}")
            else:
                logger.warning(f"Could not obtain results for {tsgym_id}")

        def get_least_loaded_gpu(candidate_gpus):
            if not candidate_gpus:
                return ''
            try:
                # Query index and memory used
                cmd = ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader']
                result = subprocess.check_output(cmd, encoding='utf-8')
                # Parse: "0, 123"
                candidates_usage = []
                for line in result.strip().splitlines():
                    if not line.strip(): continue
                    parts = line.split(',')
                    idx = parts[0].strip()
                    mem = int(parts[1].strip())
                    
                    if idx in candidate_gpus:
                        candidates_usage.append((idx, mem))
                
                if not candidates_usage:
                    # Fallback in case indices don't match or other issues
                    return candidate_gpus[0]
                
                # Sort by memory usage asc
                candidates_usage.sort(key=lambda x: x[1])
                best_gpu = candidates_usage[0][0]
                logger.info(f"Dynamic GPU Selection: Selected GPU {best_gpu} with {candidates_usage[0][1]} MiB used (Candidates: {candidates_usage})")
                return best_gpu
            except Exception as e:
                logger.warning(f"Failed to query GPU memory, falling back to random: {e}")
                import random
                return random.choice(candidate_gpus)

        def _run_script_on_gpu(script_path, gpu_candidates, target_pl):
            try:
                import tempfile
                import re as _re
                import time
                import random
                
                # Random sleep to avoid race conditions when querying GPU memory
                time.sleep(random.uniform(0.1, 2.0))
                
                # Select best GPU
                gpu_id = get_least_loaded_gpu(gpu_candidates) if isinstance(gpu_candidates, list) else gpu_candidates
                
                run_script_path = os.path.abspath(run_script)
                with open(script_path, 'r') as f:
                    content = f.read()
                content = content.replace('run.py', run_script_path)
                
                lines = content.splitlines()
                filtered_lines = []
                py_cmd_pattern = _re.compile(r'\b(python|python3|torchrun)\b')
                pl_flag_pattern = _re.compile(rf'(--pred_len|\bpred_len)\s+{_re.escape(str(target_pl))}(?!\d)') if target_pl else None

                i = 0
                while i < len(lines):
                    ln = lines[i]
                    if py_cmd_pattern.search(ln):
                        block = [ln]
                        j = i + 1
                        while j < len(lines) and lines[j].strip() != '':
                            block.append(lines[j])
                            j += 1
                        separator = [lines[j]] if j < len(lines) and lines[j].strip() == '' else []

                        include_block = True
                        if target_pl and pl_flag_pattern:
                            include_block = any(pl_flag_pattern.search(b) for b in block)

                        if include_block:
                            filtered_lines.extend(block + separator)
                        i = j + 1 if separator else j
                        continue
                    else:
                        filtered_lines.append(ln)
                        i += 1

                content = "\n".join(filtered_lines).rstrip() + "\n"
                logger.info(content)

                logger.info(f"Filtered script for {os.path.basename(script_path)} (target_pl={target_pl}, gpu={gpu_id}):\n{content}")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sh')
                tmp.write(content.encode('utf-8'))
                tmp.close()
                env = os.environ.copy()
                env['RESULTS_ROOT'] = root_path
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                subprocess.check_call(['bash', tmp.name], env=env, cwd=None)
                os.unlink(tmp.name)
                return True
            except Exception as e:
                logger.error(f"Parallel execution failed on GPU {gpu_candidates}, experiment setting:{script_path}, target_pl:{target_pl}: {e}")
                return False

        if 'to_run' in locals() and len(to_run) > 0:
            logger.info(f"Launching {len(to_run)} missing runs in parallel across GPUs: {self.cuda_devices}")
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                for idx, (script_path, tid, t_pl) in enumerate(to_run):
                    # Pass the list of all available GPUs; let the worker select the best one at runtime
                    futures.append(executor.submit(_run_script_on_gpu, script_path, self.cuda_devices, t_pl))
                concurrent.futures.wait(futures)

            logger.info(f"********************finished parallel runs, loading results...********************")
            for _, tid, t_pl in to_run:
                res_folder = None
                try:
                    res_folder = find_result_folder(tid, dataset_test, t_pl)
                except Exception:
                    logger.error(f"Failed to find result folder for {tid} on {dataset_test} with t_pl={t_pl}")
                if res_folder:
                    pred_file = os.path.join(res_folder, 'pred.npy')
                    true_file = os.path.join(res_folder, 'true.npy')
                    if os.path.exists(pred_file) and os.path.exists(true_file):
                        try:
                            pred = np.load(pred_file)
                            preds.append(pred)
                            collected_tsgym_ids.append(tid)  # Record the TSGym ID
                            if trues is None:
                                trues = np.load(true_file)
                            logger.info(f"Successfully loaded results for {tid} after parallel run. pred shape:{pred.shape}")
                        except Exception as e:
                            logger.error(f"Error loading npy files after parallel run: {e}")
                    else:
                        logger.warning(f"Files not found after run for {tid}: {pred_file}, {true_file}")

        if not preds:
            logger.warning(f"No predictions loaded for {dataset_test}")
            return np.nan, np.nan, []

        # Filter out predictions with NaNs, keeping track of corresponding TSGym IDs
        valid_preds = []
        valid_tsgym_ids = []
        for pred, tsgym_id in zip(preds, collected_tsgym_ids):
            if not np.isnan(pred).any():
                valid_preds.append(pred)
                valid_tsgym_ids.append(tsgym_id)
        
        preds = valid_preds
        collected_tsgym_ids = valid_tsgym_ids
        
        if not preds:
            logger.warning(f"No valid predictions (all contain NaN) for {dataset_test}")
            return np.nan, np.nan, []

        logger.info([_.shape for _ in preds])
        ens_pred = np.mean(np.stack(preds), axis=0)
        
        mae, mse = np.nan, np.nan
        individual_metrics = []  # List of dicts: [{'tsgym_id': str, 'mae': float, 'mse': float}, ...]
        if trues is not None:
            mae = np.mean(np.abs(ens_pred - trues))
            mse = np.mean((ens_pred - trues) ** 2)
            logger.info(f"Ensemble Top{len(preds)} Results for {dataset_test}: MAE={mae:.4f}, MSE={mse:.4f}")
            for i, (single_pred, tsgym_id) in enumerate(zip(preds, collected_tsgym_ids)):
                single_mae = np.mean(np.abs(single_pred - trues))
                single_mse = np.mean((single_pred - trues) ** 2)
                individual_metrics.append({
                    'tsgym_id': tsgym_id,
                    'mae': single_mae,
                    'mse': single_mse
                })
                logger.info(f"  Model {i+1} ({tsgym_id}) Results: MAE={single_mae:.4f}, MSE={single_mse:.4f}")
        
        return mae, mse, individual_metrics

    def meta_init(self):
        """初始化一个新的meta-learner模型"""
        n_col = [len(_) for _ in self.components.values()]
        
        if hasattr(self, 'meta_model_type') and 'icl-simple' in self.meta_model_type:
            # MetaSimpleICL: 无 Q/K/V 投影的简化版 ICL
            self.model = MetaSimpleICL(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-frozencomp':
            # MetaICLFrozenComp: 冻结组件 embedding
            self.model = MetaICLFrozenComp(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                nhead=self.nhead,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-addcomp':
            # MetaICLAddComp: 组件 embedding 加和而非拼接，维度 256
            self.model = MetaICLAddComp(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                nhead=self.nhead,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
                add_embed_dim=256,
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-labelencoder':
            # MetaICLLabelEncoder: 不使用 nn.Embedding，直接标准化后作为输入
            self.model = MetaICLLabelEncoder(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                nhead=self.nhead,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-nomasktrain-deepinput':
            # MetaICLDeepInput: 深层输入投影 (Linear->GELU->Linear->LayerNorm)
            self.model = MetaICLDeepInput(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                nhead=self.nhead,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
            self.model = MetaICLTabPFN(
                n_col=n_col,
                model_path='/data2/coding/tsgym/tabpfn-v2-regressor/tabpfn-v2-regressor-v2_default.ckpt',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        elif hasattr(self, 'meta_model_type') and self.meta_model_type.startswith('icl-'):
            self.model = MetaICL(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                nhead=self.nhead,
                num_layers=self.n_layers,
                model_type=self.meta_model_type,
                k=getattr(self, 'k', 0.0),
                temporal=getattr(self, 'temporal', 1.0),
            )
        else:
            self.model = meta_predictor(
                n_col=n_col, 
                d_model=self.d_model,
                embed_dim_meta_feature=self.meta_feature_dim, 
                dropout=self.dropout,
                n_layers=self.n_layers
            )
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(weight_decay=self.weight_decay, learning_rate=self.lr, device_type='cuda')
        self.criterion = self.loss_pearson

    def save_checkpoint(self, model_state, save_path, best_epoch=None, val_loss=None, fold_name=None):
        """
        保存meta-learner checkpoint和LabelEncoders。
        
        Args:
            model_state: 模型权重字典
            save_path: 保存路径（不含扩展名）
            best_epoch: 最佳epoch
            val_loss: 验证集loss
            fold_name: 折名称（可选）
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型checkpoint
        checkpoint = {
            'model_state_dict': model_state,
            'hyperparams': self.hyperparams,
            'best_epoch': best_epoch,
            'val_loss': val_loss,
            'fold_name': fold_name,
            'meta_model_type': self.meta_model_type,
            'components_mapping': self.components,  # 组件编码映射
            'n_col': [len(_) for _ in self.components.values()],
            'd_model': self.d_model,
            'meta_feature_dim': self.meta_feature_dim,
            'dropout': self.dropout,
            'nhead': self.nhead,
            'n_layers': self.n_layers,
            'test_dataset': self.test_dataset,
            'pred_len_1': self.pred_len_1,
            'pred_len_2': self.pred_len_2,
        }
        
        # 保存模型checkpoint (PyTorch格式)
        model_path = f"{save_path}_model.pt"
        torch.save(checkpoint, model_path)
        logger.info(f"Model checkpoint saved to: {model_path}")
        
        # 保存LabelEncoders (使用joblib)
        if hasattr(self, 'label_encoders'):
            encoders_path = f"{save_path}_label_encoders.pkl"
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"LabelEncoders saved to: {encoders_path}")
        
        return model_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path, label_encoders_path=None, device='cuda'):
        """
        加载meta-learner checkpoint和LabelEncoders进行推理。
        
        Args:
            checkpoint_path: 模型checkpoint路径 (.pt文件)
            label_encoders_path: LabelEncoders路径 (.pkl文件)，如果为None则尝试自动推断
            device: 设备
        
        Returns:
            model: 加载好的模型
            label_encoders: LabelEncoders字典
            checkpoint: checkpoint字典（包含超参数等信息）
        """
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 推断label_encoders路径
        if label_encoders_path is None:
            # 尝试从checkpoint路径推断
            base_path = checkpoint_path.replace('_model.pt', '')
            label_encoders_path = f"{base_path}_label_encoders.pkl"
        
        # 加载LabelEncoders
        label_encoders = None
        if os.path.exists(label_encoders_path):
            label_encoders = joblib.load(label_encoders_path)
            logging.info(f"LabelEncoders loaded from: {label_encoders_path}")
        else:
            logging.warning(f"LabelEncoders not found at: {label_encoders_path}")
        
        # 重建模型
        meta_model_type = checkpoint.get('meta_model_type', 'mlp')
        n_col = checkpoint['n_col']
        d_model = checkpoint['d_model']
        meta_feature_dim = checkpoint['meta_feature_dim']
        dropout = checkpoint['dropout']
        nhead = checkpoint.get('nhead', 4)
        n_layers = checkpoint.get('n_layers', 2)
        
        if 'icl-simple' in meta_model_type:
            # MetaSimpleICL: 无 Q/K/V 投影
            model = MetaSimpleICL(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                num_layers=n_layers,
                model_type=meta_model_type
            )
        elif meta_model_type == 'icl-frozencomp':
            model = MetaICLFrozenComp(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                nhead=nhead,
                num_layers=n_layers,
                model_type=meta_model_type
            )
        elif meta_model_type == 'icl-addcomp':
            model = MetaICLAddComp(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                nhead=nhead,
                num_layers=n_layers,
                model_type=meta_model_type,
                add_embed_dim=256
            )
        elif meta_model_type == 'icl-labelencoder':
            model = MetaICLLabelEncoder(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                nhead=nhead,
                num_layers=n_layers,
                model_type=meta_model_type
            )
        elif meta_model_type == 'icl-nomasktrain-deepinput':
            model = MetaICLDeepInput(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                nhead=nhead,
                num_layers=n_layers,
                model_type=meta_model_type
            )
        elif meta_model_type.startswith('icl-'):
            model = MetaICL(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                nhead=nhead,
                num_layers=n_layers,
                model_type=meta_model_type
            )
        else:
            model = meta_predictor(
                n_col=n_col,
                d_model=d_model,
                embed_dim_meta_feature=meta_feature_dim,
                dropout=dropout,
                n_layers=n_layers
            )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logging.info(f"Model loaded from: {checkpoint_path}")
        logging.info(f"  - Best epoch: {checkpoint.get('best_epoch')}")
        logging.info(f"  - Val loss: {checkpoint.get('val_loss')}")
        logging.info(f"  - Fold: {checkpoint.get('fold_name')}")
        
        return model, label_encoders, checkpoint

    def meta_fit_single_fold(self, train_data, val_data, fold_name):
        """
        训练单个折的meta-learner。
        
        Args:
            train_data: 训练数据字典
            val_data: 验证数据字典
            fold_name: 折名称（验证数据集名称）
        
        Returns:
            fold_results: 该折的训练结果
        """
        best_metric = 999
        best_epoch = 0
        es_count = 0
        es_stopped = False
        best_model_wts = copy.deepcopy(self.model.state_dict())

        # 创建DataLoader
        trainset = TensorDataset(train_data['components'], train_data['meta_features'], train_data['targets'])
        valset = TensorDataset(val_data['components'], val_data['meta_features'], val_data['targets'])
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        # 记录每个epoch的结果
        epoch_results = {
            'train_loss': [],
            'val_loss': [],
        }
        
        if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
             # Skip training for pre-trained TabPFN
             logger.info("Skipping training loop for icl-tabpfn")
             best_model_wts = {}
             best_epoch = 0
             best_metric = 0.0 # Placeholder
             es_stopped = True # Skip loop
        else:
            for epoch in tqdm(range(self.epochs), desc=f"Fold {fold_name}"):
                self.model.train()
                
                is_icl_model = hasattr(self, 'meta_model_type') and 'icl' in self.meta_model_type
                    
                if is_icl_model:
                    if self.icl_batch:
                        # Version 2: Batch Training (using trainloader with shuffle=True)
                        loss_batch = []
                        for batch in trainloader:
                            component, meta_feature, y_true = batch
                            self.optimizer.zero_grad()
                            _, y_pred = self.model(train_data=(component, meta_feature), test_data=None)
                            loss = self.criterion(y_pred.squeeze(), y_true)
                            loss.backward()
                            self.optimizer.step()
                            loss_batch.append(loss.item())
                        epoch_results['train_loss'].append(np.mean(loss_batch))
                    else:
                        # Full Batch
                        curr_c = train_data['components']
                        curr_m = train_data['meta_features']
                        curr_t = train_data['targets']
                        
                        if self.icl_shuffle:
                            # Version 1: Shuffle per epoch (Full Batch)
                            perm = torch.randperm(curr_c.size(0))
                            curr_c = curr_c[perm]
                            curr_m = curr_m[perm]
                            curr_t = curr_t[perm]
                        
                        self.optimizer.zero_grad()
                        _, y_pred = self.model((curr_c, curr_m), test_data=None)
                        loss = self.criterion(y_pred.squeeze(), curr_t.squeeze())
                        loss.backward()
                        self.optimizer.step()
                        epoch_results['train_loss'].append(loss.item())
                else:
                    loss_batch = []
                    for batch in trainloader:
                        component, meta_feature, y_true = batch
                        self.model.zero_grad()
                        _, y_pred = self.model(component, meta_feature)
                        loss = self.criterion(y_pred.squeeze(), y_true.squeeze())
                        loss.backward()
                        loss_batch.append(loss.item())
                        self.optimizer.step()
                    epoch_results['train_loss'].append(np.mean(loss_batch))
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                if hasattr(self, 'meta_model_type') and 'icl' in self.meta_model_type:
                    _, y_pred = self.model((train_data['components'], train_data['meta_features']), 
                                          test_data=(val_data['components'], val_data['meta_features']))
                    val_loss = self.criterion(y_pred.squeeze(), val_data['targets'].squeeze()).item()
                else:
                    val_preds, val_trues = [], []
                    for batch in valloader:
                        component, meta_feature, y_true = batch
                        _, y_pred = self.model(component, meta_feature)
                        val_preds.append(y_pred.squeeze())
                        val_trues.append(y_true.squeeze())
                    val_preds = torch.cat(val_preds)
                    val_trues = torch.cat(val_trues)
                    val_loss = self.criterion(val_preds, val_trues).item()
            
            epoch_results['val_loss'].append(val_loss)
            
            # 早停
            if not es_stopped:
                if val_loss < best_metric:
                    best_metric = val_loss
                    es_count = 0
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                else:
                    es_count += 1
                    if es_count > self.es_tol:
                        logger.info(f'Fold {fold_name}: Early stopping at epoch {epoch}')
                        es_stopped = True
        
        # 恢复最佳模型
        self.model.load_state_dict(best_model_wts)
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_metric,
            'epoch_results': epoch_results,
            'model_state': copy.deepcopy(best_model_wts)
        }

    @torch.no_grad()
    def meta_predict_with_model(self, model_state, train_data, return_attention=False):
        """
        使用指定的模型状态进行预测。
        
        Args:
            model_state: 模型权重
            train_data: 训练数据字典
            return_attention: 是否返回 attention scores（仅对 ICL 模型有效）
        
        Returns:
            y_preds: 预测值
            attention_info: attention 信息字典（仅当 return_attention=True 且为 ICL 模型时）
                - 'attention_scores': list of (1, L, L) Softmax Attention Scores from each layer
                - 'N_train': 训练样本数
                - 'N_test': 测试样本数
                - 四个子矩阵（按 attention 方向命名：source_to_target）：
                  - 'train_to_train': (num_layers, N_train, N_train) - TL block
                  - 'train_to_test': (num_layers, N_train, N_test) - TR block (通常被 mask)
                  - 'test_to_train': (num_layers, N_test, N_train) - BL block
                  - 'test_to_test': (num_layers, N_test, N_test) - BR block
        """
        if self.meta_model_type != 'icl-tabpfn':
            self.model.load_state_dict(model_state)
        self.model.eval()

        train_components = train_data['components']
        train_meta = train_data['meta_features']
        train_targets = train_data['targets']
        
        attention_info = None
        
        if hasattr(self, 'meta_model_type') and 'icl' in self.meta_model_type:
            kwargs = {}
            if self.meta_model_type == 'icl-tabpfn':
                kwargs['train_targets'] = train_targets

            # 对于ICL，需要构建完整的训练集作为support
            if return_attention:
                # 调用带 attention 返回的 forward
                _, y_pred, attention_info = self.model(
                    (train_components, train_meta), 
                    test_data=(self.testset_components, self.testset_meta_features),
                    return_attention=True,
                    **kwargs
                )
                # 将 attention tensors 转移到 CPU 并转为 numpy
                if attention_info is not None:
                    for key in ['train_to_train', 'train_to_test', 'test_to_train', 'test_to_test']:
                        if key in attention_info and attention_info[key] is not None:
                            attention_info[key] = attention_info[key].cpu().numpy()
            else:
                _, y_pred = self.model(
                    (train_components, train_meta), 
                    test_data=(self.testset_components, self.testset_meta_features),
                    **kwargs
                )
            y_preds = y_pred.squeeze().cpu().numpy()
        else:
            testloader = DataLoader(
                TensorDataset(self.testset_components, self.testset_meta_features, 
                             self.testset_targets, self.testset_targets_mae),
                batch_size=self.batch_size, shuffle=False, drop_last=False
            )
            y_preds = []
            for batch in testloader:
                component, meta_feature, _, _ = batch
                _, y_pred = self.model(component, meta_feature)
                y_preds.append(y_pred.squeeze())
            y_preds = torch.cat(y_preds).cpu().numpy()
        
        if return_attention:
            return y_preds, attention_info
        return y_preds

    def baseline_nn_predict(self, train_data, val_dataset):
        """
        Baseline: Nearest Neighbor 方法
        找到与测试集最近的训练数据集，使用该数据集的 performance 作为预测。
        
        Args:
            train_data: 训练数据字典 (包含多个数据集的合并数据)
            val_dataset: 验证数据集名称 (用于 logging)
        
        Returns:
            y_preds: 对测试集的预测值
            min_dist: 最近距离
        """
        # 获取测试集的 Meta Feature（假设所有测试样本属于同一个数据集，meta feature 相同）
        test_feat = self.testset_meta_features[0].cpu().numpy()  # Shape: (D_meta,)
        
        # 获取训练数据中所有样本的 Meta Features
        train_feats = train_data['meta_features'].cpu().numpy()  # Shape: (N_train, D_meta)
        train_comps = train_data['components'].cpu().numpy()  # Shape: (N_train, n_col)
        train_targets = train_data['targets'].cpu().numpy()  # Shape: (N_train,)
        
        # 计算测试集与训练集所有样本的距离 (Euclidean)
        dists = np.linalg.norm(train_feats - test_feat, axis=1)  # Shape: (N_train,)
        
        # 找到最近的样本
        min_dist = np.min(dists)
        nearest_idx = np.argmin(dists)
        nearest_feat = train_feats[nearest_idx]
        
        # 找到属于最近数据集的所有样本 (Meta Feature 相同)
        dists_to_nearest = np.linalg.norm(train_feats - nearest_feat, axis=1)
        nearest_dataset_indices = np.where(dists_to_nearest < 1e-6)[0]
        
        logger.info(f"Baseline-NN: Nearest distance={min_dist:.6f}, found {len(nearest_dataset_indices)} samples in nearest dataset.")
        
        # 构建映射: Component -> Performance (from Nearest Dataset)
        nearest_comps = train_comps[nearest_dataset_indices]
        nearest_targets = train_targets[nearest_dataset_indices]
        
        comp_perf_map = {}
        for idx, comp in enumerate(nearest_comps):
            comp_key = tuple(comp.tolist())
            comp_perf_map[comp_key] = nearest_targets[idx]
        
        # 预测测试集的 Performance
        testset_comps = self.testset_components.cpu().numpy()
        y_preds = []
        for comp in testset_comps:
            comp_key = tuple(comp.tolist())
            if comp_key in comp_perf_map:
                y_preds.append(comp_perf_map[comp_key])
            else:
                # Fallback: 使用最近数据集的平均性能
                y_preds.append(9999)
        
        return np.array(y_preds), min_dist

    def meta_fit_baseline_nn(self):
        """
        Baseline: Nearest Neighbor 方法（不需要训练）
        
        找到与 test dataset 最近的训练数据集，直接使用该数据集的 top1 组合作为预测结果。
        
        Returns:
            dict: 包含 top1 性能和名称的结果字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline: Nearest Neighbor")
        logger.info(f"Test dataset: {self.test_dataset}")
        logger.info(f"{'='*60}\n")
        
        # 获取测试集的 Meta Feature（假设同一数据集的所有样本 meta feature 相同）
        test_feat = self.testset_meta_features[0].cpu().numpy()  # Shape: (D_meta,)
        
        # 获取所有训练数据集的信息
        train_datasets = [d for d in self.datasets_train if d in self.dataset_data]
        
        logger.info(f"Training datasets: {train_datasets}")
        
        # 计算测试集与每个训练数据集的距离
        dataset_distances = {}
        for dataset in train_datasets:
            data = self.dataset_data[dataset]
            # 每个数据集的 meta feature 应该是相同的，取第一个
            dataset_feat = data['meta_features'][0]
            dist = np.linalg.norm(dataset_feat - test_feat)
            dataset_distances[dataset] = dist
            logger.info(f"  Distance to {dataset}: {dist:.6f}")
        
        # 找到最近的数据集
        nearest_dataset = min(dataset_distances, key=dataset_distances.get)
        min_dist = dataset_distances[nearest_dataset]
        
        logger.info(f"\nNearest dataset: {nearest_dataset} (distance={min_dist:.6f})")
        
        # 获取最近数据集的数据
        nearest_data = self.dataset_data[nearest_dataset]
        nearest_targets = nearest_data['targets']  # 原始 performance (MSE)
        nearest_comps = nearest_data['components']
        
        # 获取最近数据集中的所有组合，按 performance 排序（targets 越小越好）
        nearest_indices_sorted = np.argsort(nearest_targets)  # 按 performance 从小到大排序
        
        # 在测试集中查找对应的组合
        testset_comps = self.testset_components.cpu().numpy()
        y_trues = self.testset_targets.cpu().numpy()
        y_trues_mae = self.testset_targets_mae.cpu().numpy()
        
        # 构建测试集的 component -> index 映射（用于快速查找）
        testset_comp_to_idx = {}
        for idx, comp in enumerate(testset_comps):
            comp_key = tuple(comp.tolist())
            testset_comp_to_idx[comp_key] = idx
        
        # 从最近数据集的 top1 开始，顺延查找在测试集中存在的组合
        pred_top1_idx = None
        fallback_rank = 1  # 记录使用了第几优的组合
        
        for rank_idx, nearest_comp_idx in enumerate(nearest_indices_sorted):
            nearest_comp = nearest_comps[nearest_comp_idx]
            nearest_comp_key = tuple(nearest_comp.tolist())
            nearest_perf = nearest_targets[nearest_comp_idx]
            
            if nearest_comp_key in testset_comp_to_idx:
                # 找到了在测试集中存在的组合
                pred_top1_idx = testset_comp_to_idx[nearest_comp_key]
                fallback_rank = rank_idx + 1
                
                if rank_idx == 0:
                    logger.info(f"Nearest dataset top1 combination found in testset (rank={fallback_rank})")
                else:
                    logger.warning(f"Nearest dataset top{fallback_rank} combination found in testset (top1~top{rank_idx} not in testset)")
                    logger.info(f"  Top{fallback_rank} performance in nearest dataset: {nearest_perf:.6f}")
                break
            else:
                if rank_idx == 0:
                    logger.warning(f"Nearest dataset top1 combination NOT found in testset, trying next...")
                    logger.info(f"  Top1 combination: {nearest_comp_key}")
                    logger.info(f"  Top1 performance in nearest dataset: {nearest_perf:.6f}")
        
        if pred_top1_idx is not None:
            top1_perf = y_trues[pred_top1_idx]
            top1_perf_mae = y_trues_mae[pred_top1_idx]
            top1_name = self.name_components[pred_top1_idx]
            logger.info(f"Selected combination: {top1_name} (rank={fallback_rank} in nearest dataset)")
        else:
            # 如果最近数据集的所有组合都不在测试集中（这种情况应该很少见）
            logger.error(f"None of the combinations from nearest dataset found in testset!")
            logger.error(f"Nearest dataset has {len(nearest_comps)} combinations, testset has {len(testset_comps)} combinations")
            # 使用测试集中 performance 最好的组合作为 fallback
            testset_top1_idx = np.argmin(y_trues)
            top1_perf = y_trues[testset_top1_idx]
            top1_perf_mae = y_trues_mae[testset_top1_idx]
            top1_name = self.name_components[testset_top1_idx]
            fallback_rank = -1  # 表示使用了测试集的 top1 作为 fallback
            logger.warning(f"Using testset's top1 combination as fallback: {top1_name}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline-NN Results")
        logger.info(f"Nearest Dataset: {nearest_dataset}")
        logger.info(f"Used Rank in Nearest Dataset: {fallback_rank}" + (" (fallback to testset top1)" if fallback_rank == -1 else ""))
        logger.info(f"Predicted Top 1 Combination: {top1_name}")
        logger.info(f"Top 1 Performance (MSE): {top1_perf:.6f}")
        logger.info(f"Top 1 Performance (MAE): {top1_perf_mae:.6f}")
        logger.info(f"{'='*60}\n")
        
        # 保存结果
        max_size_suffix = f'-maxsize_{self.max_size}' if self.max_size else ''
        expand_suffix = f'-expand_{self.expand_testset}' if hasattr(self, 'expand_testset') else ''
        clip_suffix = f'-clip_{self.clip_timestamps}' if hasattr(self, 'clip_timestamps') else ''
        
        result_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_baseline_nn.npz'
        )
        
        # Prepend suffix if exists
        if self.suffix:
            result_path = result_path.replace(f'/{self.test_dataset}-model_', f'/{self.suffix}_{self.test_dataset}-model_')
        
        np.savez(
            result_path,
            # 基本信息
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2,
            meta_model_type=self.meta_model_type,
            
            # Baseline-NN 结果
            nearest_dataset=nearest_dataset,
            nearest_distance=min_dist,
            dataset_distances=dataset_distances,
            used_rank_in_nearest_dataset=fallback_rank,
            
            # Top1 结果
            top1_name=top1_name,
            top1_perf_mse=top1_perf,
            top1_perf_mae=top1_perf_mae,
            
            # 所有测试集真实值（用于后续分析）
            y_trues=y_trues,
            y_trues_mae=y_trues_mae,
            name_components=self.name_components,
        )
        
        logger.info(f"Results saved to: {result_path}")
        
        return {
            'nearest_dataset': nearest_dataset,
            'nearest_distance': min_dist,
            'used_rank_in_nearest_dataset': fallback_rank,
            'top1_name': top1_name,
            'top1_perf': top1_perf,
            'top1_perf_mae': top1_perf_mae,
        }

    def meta_fit_baseline_nn_dataset_ensemble(self, topk_datasets=3):
        """
        Baseline: Dataset Ensemble - 找到 top-k 最相似的数据集，使用每个数据集的 top1 组合进行 ensemble。
        
        Args:
            topk_datasets: 要查找的最相似数据集数量（默认 3）
        
        Returns:
            dict: 包含 ensemble 结果的字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline: Nearest Neighbor Dataset Ensemble (top-{topk_datasets})")
        logger.info(f"Test dataset: {self.test_dataset}")
        logger.info(f"{'='*60}\n")
        
        # 获取测试集的 Meta Feature
        test_feat = self.testset_meta_features[0].cpu().numpy()
        
        # 获取所有训练数据集的信息
        train_datasets = [d for d in self.datasets_train if d in self.dataset_data]
        logger.info(f"Training datasets: {train_datasets}")
        
        # 计算测试集与每个训练数据集的距离
        dataset_distances = {}
        for dataset in train_datasets:
            data = self.dataset_data[dataset]
            dataset_feat = data['meta_features'][0]
            dist = np.linalg.norm(dataset_feat - test_feat)
            dataset_distances[dataset] = dist
            logger.info(f"  Distance to {dataset}: {dist:.6f}")
        
        # 找到 top-k 最近的数据集
        sorted_datasets = sorted(dataset_distances.items(), key=lambda x: x[1])
        topk_nearest = sorted_datasets[:topk_datasets]
        
        logger.info(f"\nTop-{topk_datasets} nearest datasets:")
        for i, (ds, dist) in enumerate(topk_nearest):
            logger.info(f"  {i+1}. {ds} (distance={dist:.6f})")
        
        # 收集每个最近数据集的 top1 组合名称
        topk_names = []
        testset_comps = self.testset_components.cpu().numpy()
        y_trues = self.testset_targets.cpu().numpy()
        y_trues_mae = self.testset_targets_mae.cpu().numpy()
        
        # 构建测试集的 component -> index 映射
        testset_comp_to_idx = {}
        for idx, comp in enumerate(testset_comps):
            comp_key = tuple(comp.tolist())
            testset_comp_to_idx[comp_key] = idx
        
        for ds_name, ds_dist in topk_nearest:
            nearest_data = self.dataset_data[ds_name]
            nearest_targets = nearest_data['targets']
            nearest_comps = nearest_data['components']
            nearest_indices_sorted = np.argsort(nearest_targets)
            
            # 找到在测试集中存在的最佳组合
            found = False
            for rank_idx, nearest_comp_idx in enumerate(nearest_indices_sorted):
                nearest_comp = nearest_comps[nearest_comp_idx]
                nearest_comp_key = tuple(nearest_comp.tolist())
                
                if nearest_comp_key in testset_comp_to_idx:
                    test_idx = testset_comp_to_idx[nearest_comp_key]
                    combo_name = self.name_components[test_idx]
                    topk_names.append(combo_name)
                    logger.info(f"  From {ds_name}: selected '{combo_name}' (rank {rank_idx+1} in source dataset)")
                    found = True
                    break
            
            if not found:
                logger.warning(f"  From {ds_name}: no matching combination found in testset")
        
        if not topk_names:
            logger.error("No valid combinations found from any nearest dataset!")
            return {'ensemble_mae': np.nan, 'ensemble_mse': np.nan, 'topk_names': []}
        
        logger.info(f"\nSelected combinations for ensemble: {topk_names}")
        
        # 使用现有的 ensemble_predictions 方法进行集成
        ens_mae, ens_mse, individual_metrics = self.ensemble_predictions(
            topk_names, self.test_dataset, root_path=self.write_results_root
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline-NN Dataset Ensemble Results")
        logger.info(f"Ensemble MAE: {ens_mae:.6f}" if not np.isnan(ens_mae) else "Ensemble MAE: N/A")
        logger.info(f"Ensemble MSE: {ens_mse:.6f}" if not np.isnan(ens_mse) else "Ensemble MSE: N/A")
        logger.info(f"{'='*60}\n")
        
        # 保存结果
        max_size_suffix = f'-maxsize_{self.max_size}' if self.max_size else ''
        expand_suffix = f'-expand_{self.expand_testset}' if hasattr(self, 'expand_testset') else ''
        clip_suffix = f'-clip_{self.clip_timestamps}' if hasattr(self, 'clip_timestamps') else ''
        
        result_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_baseline_nn_dataset_ensemble.npz'
        )
        
        if self.suffix:
            result_path = result_path.replace(f'/{self.test_dataset}-model_', f'/{self.suffix}_{self.test_dataset}-model_')
        
        np.savez(
            result_path,
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2,
            meta_model_type=self.meta_model_type,
            topk_nearest_datasets=[ds for ds, _ in topk_nearest],
            dataset_distances=dataset_distances,
            topk_names=topk_names,
            ensemble_mae=ens_mae,
            ensemble_mse=ens_mse,
            individual_metrics=individual_metrics,
            y_trues=y_trues,
            y_trues_mae=y_trues_mae,
            name_components=self.name_components,
        )
        
        logger.info(f"Results saved to: {result_path}")
        
        return {
            'topk_nearest_datasets': [ds for ds, _ in topk_nearest],
            'topk_names': topk_names,
            'ensemble_mae': ens_mae,
            'ensemble_mse': ens_mse,
            'individual_metrics': individual_metrics,
        }

    def meta_fit_baseline_nn_components_ensemble(self, topk_components=5):
        """
        Baseline: Components Ensemble - 找到 top-1 最相似的数据集，使用该数据集的 top-k 组合进行 ensemble。
        
        Args:
            topk_components: 要使用的组合数量（默认 5）
        
        Returns:
            dict: 包含 ensemble 结果的字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline: Nearest Neighbor Components Ensemble (top-{topk_components} from nearest)")
        logger.info(f"Test dataset: {self.test_dataset}")
        logger.info(f"{'='*60}\n")
        
        # 获取测试集的 Meta Feature
        test_feat = self.testset_meta_features[0].cpu().numpy()
        
        # 获取所有训练数据集的信息
        train_datasets = [d for d in self.datasets_train if d in self.dataset_data]
        logger.info(f"Training datasets: {train_datasets}")
        
        # 计算测试集与每个训练数据集的距离
        dataset_distances = {}
        for dataset in train_datasets:
            data = self.dataset_data[dataset]
            dataset_feat = data['meta_features'][0]
            dist = np.linalg.norm(dataset_feat - test_feat)
            dataset_distances[dataset] = dist
            logger.info(f"  Distance to {dataset}: {dist:.6f}")
        
        # 找到最近的数据集
        nearest_dataset = min(dataset_distances, key=dataset_distances.get)
        min_dist = dataset_distances[nearest_dataset]
        logger.info(f"\nNearest dataset: {nearest_dataset} (distance={min_dist:.6f})")
        
        # 获取最近数据集的数据
        nearest_data = self.dataset_data[nearest_dataset]
        nearest_targets = nearest_data['targets']
        nearest_comps = nearest_data['components']
        nearest_indices_sorted = np.argsort(nearest_targets)
        
        testset_comps = self.testset_components.cpu().numpy()
        y_trues = self.testset_targets.cpu().numpy()
        y_trues_mae = self.testset_targets_mae.cpu().numpy()
        
        # 构建测试集的 component -> index 映射
        testset_comp_to_idx = {}
        for idx, comp in enumerate(testset_comps):
            comp_key = tuple(comp.tolist())
            testset_comp_to_idx[comp_key] = idx
        
        # 收集 top-k 组合名称
        topk_names = []
        for rank_idx, nearest_comp_idx in enumerate(nearest_indices_sorted):
            if len(topk_names) >= topk_components:
                break
            
            nearest_comp = nearest_comps[nearest_comp_idx]
            nearest_comp_key = tuple(nearest_comp.tolist())
            nearest_perf = nearest_targets[nearest_comp_idx]
            
            if nearest_comp_key in testset_comp_to_idx:
                test_idx = testset_comp_to_idx[nearest_comp_key]
                combo_name = self.name_components[test_idx]
                topk_names.append(combo_name)
                logger.info(f"  Rank {rank_idx+1} in {nearest_dataset}: '{combo_name}' (perf={nearest_perf:.6f})")
            else:
                logger.warning(f"  Rank {rank_idx+1} in {nearest_dataset}: not found in testset, skipping")
        
        if not topk_names:
            logger.error("No valid combinations found from nearest dataset!")
            return {'ensemble_mae': np.nan, 'ensemble_mse': np.nan, 'topk_names': []}
        
        logger.info(f"\nSelected top-{len(topk_names)} combinations from {nearest_dataset}: {topk_names}")
        
        # 使用现有的 ensemble_predictions 方法进行集成
        ens_mae, ens_mse, individual_metrics = self.ensemble_predictions(
            topk_names, self.test_dataset, root_path=self.write_results_root
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Baseline-NN Components Ensemble Results")
        logger.info(f"Nearest Dataset: {nearest_dataset}")
        logger.info(f"Number of Combinations: {len(topk_names)}")
        logger.info(f"Ensemble MAE: {ens_mae:.6f}" if not np.isnan(ens_mae) else "Ensemble MAE: N/A")
        logger.info(f"Ensemble MSE: {ens_mse:.6f}" if not np.isnan(ens_mse) else "Ensemble MSE: N/A")
        logger.info(f"{'='*60}\n")
        
        # 保存结果
        max_size_suffix = f'-maxsize_{self.max_size}' if self.max_size else ''
        expand_suffix = f'-expand_{self.expand_testset}' if hasattr(self, 'expand_testset') else ''
        clip_suffix = f'-clip_{self.clip_timestamps}' if hasattr(self, 'clip_timestamps') else ''
        
        result_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_baseline_nn_components_ensemble.npz'
        )
        
        if self.suffix:
            result_path = result_path.replace(f'/{self.test_dataset}-model_', f'/{self.suffix}_{self.test_dataset}-model_')
        
        np.savez(
            result_path,
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2,
            meta_model_type=self.meta_model_type,
            nearest_dataset=nearest_dataset,
            nearest_distance=min_dist,
            dataset_distances=dataset_distances,
            topk_names=topk_names,
            ensemble_mae=ens_mae,
            ensemble_mse=ens_mse,
            individual_metrics=individual_metrics,
            y_trues=y_trues,
            y_trues_mae=y_trues_mae,
            name_components=self.name_components,
        )
        
        logger.info(f"Results saved to: {result_path}")
        
        return {
            'nearest_dataset': nearest_dataset,
            'nearest_distance': min_dist,
            'topk_names': topk_names,
            'ensemble_mae': ens_mae,
            'ensemble_mse': ens_mse,
            'individual_metrics': individual_metrics,
        }

    def meta_fit_fewshot_best(self):
        """
        Meta-Learner: Fewshot-Best
        Directly select the best performance from the few-shot results of the test dataset.
        This serves as an Oracle/Upper-bound baseline using few-shot data.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Meta-Learner: Fewshot-Best")
        logger.info(f"Test dataset: {self.test_dataset}")
        logger.info(f"{'='*60}\n")
        
        result_path_fewshot = './results_long_term_forecasting_fewshot/resultsGym_MLP'
        d_path = os.path.join(result_path_fewshot, self.test_dataset)
        
        if not os.path.exists(d_path):
            logger.error(f"Fewshot results directory not found: {d_path}")
            return {'mse': np.nan, 'mae': np.nan}
            
        # Scan all results
        best_mse = float('inf')
        best_mae = float('inf')
        best_name = None
        best_file = None
        
        files = os.listdir(d_path)
        logger.info(f"Scanning {len(files)} few-shot results...")
        
        for f in files:
            # Check Pred Len
            if self.arg_all_periods:
                # If checking all periods, we need to match current logic? 
                # Usually we want the best average or best for specific pl?
                # For simplicity, assuming we look for Best MSE across whatever is valid.
                pass
            else:
                 # Check if file corresponds to target pred_len
                 target_pl = self.pred_len_2 if self.test_dataset in ['ili', 'nyse', 'nasdaq'] else self.pred_len_1
                 if f'pl{target_pl}' not in f:
                     continue
            
            metric_path = os.path.join(d_path, f, 'metrics.npy')
            if not os.path.exists(metric_path):
                continue
                
            try:
                metrics = np.load(metric_path)
                mae = metrics[0]
                mse = metrics[1]
                
                if np.isnan(mse): continue
                
                if mse < best_mse:
                    best_mse = mse
                    best_mae = mae
                    best_name = f
                    best_file = metric_path
            except Exception as e:
                continue
                
        if not best_name:
            logger.error("No valid few-shot results found.")
            return {'mse': np.nan, 'mae': np.nan}
            
        logger.info(f"Best Fewshot Configuration Found: {best_name}")
        logger.info(f"Fewshot Metrics - MSE: {best_mse:.6f}, MAE: {best_mae:.6f}")
        
        # Now retrieve the Full-Shot result for this configuration
        # Construct path to full-shot result
        # Full-shot results are in self.read_results_root/resultsGym_MLP/dataset/filename
        # best_name is the filename (directory name)
        
        full_shot_root = self.read_results_root  # e.g., ./results_long_term_forecasting
        # Assuming MLP backbone for now as fewshot path was hardcoded to MLP
        full_shot_path = os.path.join(full_shot_root, 'resultsGym_MLP', self.test_dataset, best_name)
        full_shot_metric_file = os.path.join(full_shot_path, 'metrics.npy')
        
        full_mse, full_mae = np.nan, np.nan
        
        if os.path.exists(full_shot_metric_file):
            try:
                fs_metrics = np.load(full_shot_metric_file)
                full_mae = fs_metrics[0]
                full_mse = fs_metrics[1]
                logger.info(f"Full-Shot Metrics Retrieved for {best_name}")
                logger.info(f"MSE: {full_mse:.6f}, MAE: {full_mae:.6f}")
            except Exception as e:
                logger.error(f"Error loading full-shot metrics from {full_shot_metric_file}: {e}")
        else:
            logger.warning(f"Full-shot results not found for configuration {best_name} at {full_shot_path}")
            # Optional: Run the full-shot script if missing? 
            # For now, we return NaNs as per typical evaluation logic or relying on pre-computed results.
            
        
        # Save Results
        max_size_suffix = f'-maxsize_{self.max_size}' if self.max_size else ''
        hp_suffix = f'-fewshot_best'
        
        save_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.suffix + "_" if self.suffix else ""}{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'{max_size_suffix}{hp_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_fewshot_best.npz'
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # We save minimal info since this is a baseline
        np.savez_compressed(
            save_path,
            top1_perf=full_mse,
            top1_perf_mae=full_mae,
            top_name=best_name,
            full_path=full_shot_metric_file,
            fewshot_mse=best_mse,
            fewshot_mae=best_mae,
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2
        )
        
        logger.info(f"Fewshot-Best results saved to: {save_path}")

        return {
            'mse': full_mse,
            'mae': full_mae,
            'top_name': best_name,
            'full_path': full_shot_metric_file,
            'fewshot_mse': best_mse,
            'fewshot_mae': best_mae
        }

    # =================== Legacy模式评估方法（用于arg_all_periods模式） ===================
    def _evaluate_legacy_with_trained_model(self, model_state, train_history=None):
        """
        使用已训练的模型在当前测试集上进行评估。
        用于 arg_all_periods=True 且 use_kfold=False (legacy模式) 时，
        训练一次后对不同 pred_len 的测试集分别评估。
        
        Args:
            model_state: 已训练模型的 state_dict
            train_history: Optional training history to save with results
        
        Returns:
            dict: 评估结果
        """
        self.model.load_state_dict(model_state)
        logger.info(f"[Legacy] Evaluating with trained model on test set (pred_len={self.pred_len_1}/{self.pred_len_2})")
        
        # 加载模型权重
        self.model.eval()
        
        # 获取测试集真实值
        y_trues = self.testset_targets.cpu().numpy()
        y_trues_mae = self.testset_targets_mae.cpu().numpy()
        has_valid_targets = not np.isnan(y_trues).all()
        
        is_icl_model = hasattr(self, 'meta_model_type') and 'icl' in self.meta_model_type
        
        with torch.no_grad():
            if is_icl_model:
                # ICL模型：使用训练数据作为context
                all_train_components = []
                all_train_meta = []
                for dataset, data in self.dataset_data.items():
                    if dataset != self.test_dataset:
                        all_train_components.append(data['components'])
                        all_train_meta.append(data['meta_features'])
                
                train_comp = torch.from_numpy(np.concatenate(all_train_components, axis=0)).long().to(self.device)
                train_meta_tensor = torch.from_numpy(np.concatenate(all_train_meta, axis=0)).float().to(self.device)
                
                _, y_pred = self.model(
                    (train_comp, train_meta_tensor), 
                    test_data=(self.testset_components, self.testset_meta_features)
                )
                y_preds = y_pred.squeeze().cpu().numpy()
            else:
                # MLP模型：直接在测试集上预测
                testloader = DataLoader(
                    TensorDataset(self.testset_components, self.testset_meta_features, self.testset_targets, self.testset_targets_mae),
                    batch_size=self.batch_size, shuffle=False, drop_last=False
                )
                y_preds_list = []
                for batch in testloader:
                    component, meta_feature, _, _ = batch
                    _, y_pred = self.model(component, meta_feature)
                    y_preds_list.append(y_pred.squeeze())
                y_preds = torch.cat(y_preds_list).cpu().numpy()
        
        # 计算评估指标
        if not has_valid_targets:
            top1_perf = np.nan
            top1_perf_mae = np.nan
            pred_ranks_for_true_topk = np.nan
            true_ranks_for_pred_topk = np.nan
        else:
            top1_perf = y_trues[np.argmin(y_preds)]
            top1_perf_mae = y_trues_mae[np.argmin(y_preds)]

            pred_ranks = np.argsort(np.argsort(y_preds)) + 1
            true_ranks = np.argsort(np.argsort(y_trues)) + 1

            topk = 5
            true_topk_indices = np.argsort(y_trues)[:topk]
            pred_topk_indices = np.argsort(y_preds)[:topk]
            
            pred_ranks_for_true_topk = np.mean(pred_ranks[true_topk_indices])
            true_ranks_for_pred_topk = np.mean(true_ranks[pred_topk_indices])
        
        # 获取top名称
        top_name = self.name_components[np.argmin(y_preds)]
        topk_names = [self.name_components[i] for i in np.argsort(y_preds)[:5]]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[Legacy Evaluation] Results for pred_len={self.pred_len_1}/{self.pred_len_2}")
        logger.info(f"Top 1 Combination: {top_name}")
        logger.info(f"Top 5 Combinations: {topk_names}")
        if has_valid_targets:
            logger.info(f"Top 1 MSE: {top1_perf:.6f}, Top 1 MAE: {top1_perf_mae:.6f}")
            logger.info(f"Pred ranks for true top5: {pred_ranks_for_true_topk:.2f}")
            logger.info(f"True ranks for pred top5: {true_ranks_for_pred_topk:.2f}")
        logger.info(f"{'='*60}\n")
        
        # 保存结果（简化版，不运行 ensemble）
        max_size_suffix = f'-max_size_{self.max_size}' if self.max_size is not None else ''
        expand_suffix = f'-expand_testset' if self.expand_testset else ''
        clip_suffix = f'-clip_{self.clip_timestamps}{int(self.cutoff_time)}' if self.clip_timestamps else ''
        
        icl_suffix = f'-ishuf_{self.icl_shuffle}-ibatch_{self.icl_batch}' if hasattr(self, 'icl_shuffle') else ''
        k_val = getattr(self, "k", 0.0)
        temp_val = getattr(self, "temporal", 1.0)
        if "icl" in getattr(self, "meta_model_type", ""):
            ktemp_suffix = f"-k_{k_val}-temp_{temp_val}"
        else:
            ktemp_suffix = ""
        hp_suffix = f'-lr_{self.lr}-dm_{self.d_model}-nl_{self.n_layers}-wd_{self.weight_decay}{icl_suffix}{ktemp_suffix}'
        
        save_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.suffix + "_" if self.suffix else ""}{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'-ensemble_{self.ensemble_enabled}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}{hp_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_legacy_eval.npz'
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez_compressed(
            save_path,
            # 超参数
            hyperparams=self.hyperparams,
            
            # 评估模式标识
            k_folds=1,
            training_mode='legacy_all_periods_eval',
            
            # 预测结果
            predictions=y_preds,
            pred_ranks_for_true_topk=pred_ranks_for_true_topk,
            true_ranks_for_pred_topk=true_ranks_for_pred_topk,
            total_num=len(y_preds),
            top1_perf=top1_perf,
            top1_perf_mae=top1_perf_mae,
            top_name=top_name,
            topk_names=topk_names,
            
            # 其他
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2,
            epoch_metrics=train_history.get('epoch_metrics', []) if train_history else []
        )
        
        logger.info(f"[Legacy Evaluation] Results saved to: {save_path}")
        
        return {
            'predictions': y_preds,
            'top_name': top_name,
            'topk_names': topk_names,
            'top1_perf': top1_perf,
            'top1_perf_mae': top1_perf_mae,
        }

    # =================== 保留原有的单模型训练方法（向后兼容） ===================
    def meta_fit(self, best_metric=999, best_epoch=0, es_count=0, es_tol=5, es_stopped=False):
        """原有的单模型训练方法，保留用于向后兼容"""
        logger.info("Warning: Using legacy single-model training.")
        
        # 构建完整训练集（对每个数据集内部分别做 rank）
        all_train_components = []
        all_train_meta = []
        all_train_targets_rank = []
        all_train_targets_metrics = []
        
        for dataset, data in self.dataset_data.items():
            if dataset != self.test_dataset:
                all_train_components.append(data['components'])
                all_train_meta.append(data['meta_features'])
                # 对每个数据集内部分别做 rank（归一化到 [1/n, 1]）
                targets = data['targets']
                targets_rank = (np.argsort(np.argsort(targets)).astype(np.float32) + 1) / len(targets)
                lower, upper = np.percentile(targets, [0.1, 99.9])
                clipped = np.clip(targets, lower, upper)
                targets = (clipped - np.mean(clipped)) / np.std(clipped)
                all_train_targets_metrics.append(targets)
                all_train_targets_rank.append(targets_rank)

        
        train_components = np.concatenate(all_train_components, axis=0)
        train_meta = np.concatenate(all_train_meta, axis=0)
        train_targets_rank = np.concatenate(all_train_targets_rank, axis=0)
        train_targets_metrics = np.concatenate(all_train_targets_metrics, axis=0)
        
        train_components = torch.from_numpy(train_components).long().to(self.device)
        train_meta = torch.from_numpy(train_meta).float().to(self.device)
        train_targets = torch.from_numpy(train_targets_rank).float().to(self.device)
        if self.suffix == 'rawlabel':
            train_targets = torch.from_numpy(train_targets_metrics).float().to(self.device)
        
        # 70/30分割
        train_size = int(0.7 * len(train_targets))
        indices = torch.randperm(len(train_targets))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        print(f"train target shape:{train_targets.shape}, train components shape:{train_components.shape}, train meta shape:{train_meta.shape}")
        # val compoents
        val_components = train_components[val_indices,:]
        val_meta = train_meta[val_indices,:]
        val_targets = train_targets[val_indices]

        # train compoents
        train_components = train_components[train_indices,:]
        train_meta = train_meta[train_indices,:]
        train_targets = train_targets[train_indices]

        trainset = TensorDataset(train_components, train_meta, train_targets)
        valset = TensorDataset(val_components, val_meta, val_targets)
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        epoch_metrics_history = []
        is_icl_model = hasattr(self, 'meta_model_type') and 'icl' in self.meta_model_type
        
        if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
             logger.info("Skipping training loop for icl-tabpfn (legacy mode)")
             best_epoch = 0
             best_metric = 0.0
             es_stopped = True
        else:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                loss_batch = []
                if is_icl_model:
                    if self.icl_batch:
                        # Version 2: Batch Training (using trainloader with shuffle=True)
                        for batch in trainloader:
                            component, meta_feature, y_true = batch
                            self.optimizer.zero_grad()
                            _, y_pred = self.model(train_data=(component, meta_feature), test_data=None)
                            loss = self.criterion(y_pred.squeeze(), y_true)
                            loss.backward()
                            self.optimizer.step()
                            loss_batch.append(loss.item())
                    else:
                        # Full Batch
                        curr_c, curr_m, curr_t = train_components, train_meta, train_targets
                        
                        if self.icl_shuffle:
                            # Version 1: Shuffle per epoch (Full Batch)
                            perm = torch.randperm(curr_c.size(0))
                            curr_c = curr_c[perm]
                            curr_m = curr_m[perm]
                            curr_t = curr_t[perm]
                        
                        self.optimizer.zero_grad()
                        _, y_pred = self.model(train_data=(curr_c, curr_m), test_data=None)
                        loss = self.criterion(y_pred.squeeze(), curr_t)
                        loss.backward()
                        self.optimizer.step()
                        loss_batch.append(loss.item())
                else:
                    for batch in trainloader:
                        component, meta_feature, y_true = batch
                        self.model.zero_grad()
                        _, y_pred = self.model(component, meta_feature)
                        loss = self.criterion(y_pred.squeeze(), y_true.squeeze())
                        loss.backward()
                        loss_batch.append(loss.item())
                        self.optimizer.step()

                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_preds, val_trues = [], []
                    if not is_icl_model:
                        # Non-ICL standard epoch validation
                        val_preds, val_trues = [], []
                        for batch in valloader:
                            component, meta_feature, y_true = batch
                            _, y_pred = self.model(component, meta_feature)
                            val_preds.append(y_pred.view(-1))
                            val_trues.append(y_true.view(-1))
                        # print(val_preds)
                        # print(torch.cat(val_preds).flatten())
                        # print(torch.cat(val_trues).flatten())
                        val_loss = self.criterion(torch.cat(val_preds).flatten(), torch.cat(val_trues).flatten()).item()
                    else:
                        if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
                            val_loss = 0.0
                        elif self.icl_batch:
                            val_preds, val_trues = [], []
                            for batch in valloader:
                                component, meta_feature, y_true = batch
                                _, y_pred = self.model(
                                    train_data=(train_components, train_meta),
                                    test_data=(component, meta_feature)
                                )
                                val_preds.append(y_pred.view(-1))
                                val_trues.append(y_true.view(-1))
                            val_loss = self.criterion(torch.cat(val_preds).flatten(), torch.cat(val_trues).flatten()).item()
                        else:
                            # Standard ICL Validation
                            _, y_pred = self.model(train_data=(train_components, train_meta), test_data=(val_components, val_meta))
                            val_loss = self.criterion(y_pred.squeeze(), val_targets).item()
                
                # Record metrics per epoch
                current_epoch_metrics = {'epoch': epoch, 'val_loss': val_loss}
                all_eval_datasets = list(self.dataset_data.keys())
                if self.test_dataset not in all_eval_datasets:
                    pass
                
                for d_name in all_eval_datasets + [self.test_dataset]:
                    # Avoid dupes if test_dataset is in dataset_data
                    if d_name == self.test_dataset and d_name in all_eval_datasets and 'test_dataset' not in current_epoch_metrics:
                    # Test Dataset Logic
                    # Use self.testset_components
                        c = self.testset_components
                        m = self.testset_meta_features
                        t_mse = self.testset_targets.cpu().numpy()
                        t_mae = self.testset_targets_mae.cpu().numpy()
                        
                        if is_icl_model:
                            # ICL requires context. Use full training data as context.
                            kwargs = {}
                            if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
                                kwargs['train_targets'] = train_targets
                            
                            _, y_pred = self.model(
                                train_data=(train_components, train_meta),
                                test_data=(c, m),
                                **kwargs
                            )
                        else:
                            # MLP
                            _, y_pred = self.model(c, m)
                        
                        y_pred = y_pred.squeeze().detach().cpu().numpy()
                        
                        # Top 1
                        best_idx = np.argmin(y_pred)
                        
                        # Rank in ground truth
                        # Ranks are per dataset. Low MSE = Rank 1.
                        true_ranks = np.argsort(np.argsort(t_mse)) + 1
                        sel_rank = true_ranks[best_idx]
                        sel_mse = t_mse[best_idx]
                        sel_mae = t_mae[best_idx]
                        
                        current_epoch_metrics[d_name] = {
                            'rank': float(sel_rank),
                            'mse': float(sel_mse),
                            'mae': float(sel_mae)
                        }
                        continue
                    
                    if d_name == self.test_dataset: continue # Handled above
                    
                    # Training Datasets
                    d_data = self.dataset_data[d_name]
                    c = torch.from_numpy(d_data['components']).long().to(self.device)
                    m = torch.from_numpy(d_data['meta_features']).float().to(self.device)
                    t_mse = d_data['targets'] # Raw MSE
                    # No MAE available in dataset_data currently for train sets.
                    
                    if is_icl_model:
                        # Context? Using self as context might be weird for ICL?
                        # Typically we use "train_components" (mixed) as context.
                        kwargs = {}
                        if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
                            kwargs['train_targets'] = train_targets

                        _, y_pred = self.model(
                                 train_data=(train_components, train_meta),
                                 test_data=(c, m),
                                 **kwargs
                             )
                    else:
                        _, y_pred = self.model(c, m)
                    
                    y_pred = y_pred.squeeze().detach().cpu().numpy()
                    best_idx = np.argmin(y_pred)
                    
                    true_ranks = np.argsort(np.argsort(t_mse)) + 1
                    sel_rank = true_ranks[best_idx]
                    sel_mse = t_mse[best_idx]
                    
                    current_epoch_metrics[d_name] = {
                        'rank': float(sel_rank),
                        'mse': float(sel_mse),
                        'mae': np.nan # Not available
                    }
                
                epoch_metrics_history.append(current_epoch_metrics)

                if not es_stopped:
                    if val_loss < best_metric:
                        best_metric = val_loss
                        es_count = 0
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                    else:
                        es_count += 1
                        if es_count > es_tol:
                            logger.info(f'Early stopping at epoch: {epoch}')
                            es_stopped = True
        
        if hasattr(self, 'meta_model_type') and self.meta_model_type != 'icl-tabpfn':
            self.model.load_state_dict(best_model_wts)
        logger.info(f'Best Epoch: {best_epoch}')
        
        # Log Epoch History
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch-wise Top-1 Selection Performance for {self.test_dataset} with predlen {self.pred_len_1}/{self.pred_len_2}")
        logger.info(f"{'Epoch':<6} | {'Test Rank':<10} | {'Test MSE':<10} | {'Test MAE':<10} | {'Val Loss':<10}")
        for em in epoch_metrics_history:
            test_m = em.get(self.test_dataset, {})
            t_rank = f"{test_m.get('rank', 'N/A'):.1f}"
            t_mse = f"{test_m.get('mse', 'N/A'):.4f}"
            t_mae = f"{test_m.get('mae', 'N/A'):.4f}"
            v_loss = f"{em.get('val_loss', 'N/A'):.4f}"
            logger.info(f"{em['epoch']:<6} | {t_rank:<10} | {t_mse:<10} | {t_mae:<10} | {v_loss:<10}")
        logger.info(f"{'='*60}\n")
        
        # =================== 在测试集上预测 ===================
        self.model.eval()
        y_trues = self.testset_targets.cpu().numpy()
        y_trues_mae = self.testset_targets_mae.cpu().numpy()
        has_valid_targets = not np.isnan(y_trues).all()
        
        # 检查是否需要保存 attention
        should_save_attention = self.save_attention and is_icl_model
        attention_info = None
        
        with torch.no_grad():
            if is_icl_model:
                # ICL模型：使用训练数据作为context
                all_train_components = []
                all_train_meta = []
                for dataset, data in self.dataset_data.items():
                    if dataset != self.test_dataset:
                        all_train_components.append(data['components'])
                        all_train_meta.append(data['meta_features'])
                
                train_comp = torch.from_numpy(np.concatenate(all_train_components, axis=0)).long().to(self.device)
                train_meta_tensor = torch.from_numpy(np.concatenate(all_train_meta, axis=0)).float().to(self.device)
                
                if should_save_attention:
                    # 获取 attention scores
                    _, y_pred, attention_info = self.model(
                        (train_comp, train_meta_tensor), 
                        test_data=(self.testset_components, self.testset_meta_features),
                        return_attention=True
                    )
                    # 将 attention tensors 转移到 CPU 并转为 numpy
                    if attention_info is not None:
                        for key in ['train_to_train', 'train_to_test', 'test_to_train', 'test_to_test']:
                            if key in attention_info and attention_info[key] is not None:
                                attention_info[key] = attention_info[key].cpu().numpy()
                else:
                    kwargs = {}
                    if hasattr(self, 'meta_model_type') and self.meta_model_type == 'icl-tabpfn':
                        # For TabPFN, we need the targets corresponding to the Full context (all_train_components).
                        # The variable 'train_targets' in this scope is the 70% training subset (from line 3916),
                        # which does NOT match 'train_comp' (full set) constructed here.
                        # We must reconstruct the full targets consistent with all_train_actions.
                        
                        if self.suffix == 'rawlabel':
                             # Use the raw/metric targets if rawlabel is specified
                             full_targets_np = train_targets_metrics
                        else:
                             # Use the ranked targets
                             full_targets_np = train_targets_rank
                        
                        # Ensure it is on the correct device
                        kwargs['train_targets'] = torch.from_numpy(full_targets_np).float().to(self.device)
                    
                    _, y_pred = self.model(
                        (train_comp, train_meta_tensor), 
                        test_data=(self.testset_components, self.testset_meta_features),
                        **kwargs
                    )
                y_preds = y_pred.squeeze().cpu().numpy()
            else:
                # MLP模型：直接在测试集上预测
                testloader = DataLoader(
                    TensorDataset(self.testset_components, self.testset_meta_features, self.testset_targets, self.testset_targets_mae),
                    batch_size=self.batch_size, shuffle=False, drop_last=False
                )
                y_preds_list = []
                for batch in testloader:
                    component, meta_feature, _, _ = batch
                    _, y_pred = self.model(component, meta_feature)
                    y_preds_list.append(y_pred.view(-1))
                y_preds = torch.cat(y_preds_list).cpu().numpy()
        
        # =================== 计算评估指标 ===================
        if not has_valid_targets:
            top1_perf = np.nan
            top1_perf_mae = np.nan
            pred_ranks_for_true_topk = np.nan
            true_ranks_for_pred_topk = np.nan
        else:
            top1_perf = y_trues[np.argmin(y_preds)]
            top1_perf_mae = y_trues_mae[np.argmin(y_preds)]

            pred_ranks = np.argsort(np.argsort(y_preds)) + 1
            true_ranks = np.argsort(np.argsort(y_trues)) + 1

            topk = 5
            true_topk_indices = np.argsort(y_trues)[:topk]
            pred_topk_indices = np.argsort(y_preds)[:topk]
            
            pred_ranks_for_true_topk = np.mean(pred_ranks[true_topk_indices])
            true_ranks_for_pred_topk = np.mean(true_ranks[pred_topk_indices])
        
        # 获取top名称
        top_name = self.name_components[np.argmin(y_preds)]
        topk_names = [self.name_components[i] for i in np.argsort(y_preds)[:5]]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Legacy Single-Model Results")
        logger.info(f"Top 1 Combination: {top_name}")
        logger.info(f"Top 5 Combinations: {topk_names}")
        if has_valid_targets:
            logger.info(f"Top 1 MSE: {top1_perf:.6f}, Top 1 MAE: {top1_perf_mae:.6f}")
            logger.info(f"Pred ranks for true top5: {pred_ranks_for_true_topk:.2f}")
            logger.info(f"True ranks for pred top5: {true_ranks_for_pred_topk:.2f}")
        logger.info(f"Best training epoch: {best_epoch}")
        logger.info(f"{'='*60}\n")
        
        # =================== 集成top5组合或expand testset ===================
        ens_mae, ens_mse, individual_metrics = np.nan, np.nan, []
        selected_script_metrics = {'mae': np.nan, 'mse': np.nan, 'script': None, 'result_folder': None}
        
        if self.expand_testset:
            # expand模式：选择预测的top1，运行实验
            best_name = topk_names[0] if len(topk_names) > 0 else None
            if best_name and hasattr(self, 'test_script_map') and best_name in self.test_script_map:
                script_path = self.test_script_map[best_name]
                selected_script_metrics['script'] = script_path
                
                import glob
                match = re.search(r'(TSGym\d+)', best_name)
                tid = match.group(1) if match else None
                pl_match = re.search(r'_pl(\d+)\.sh$', os.path.basename(script_path))
                target_pl = pl_match.group(1) if pl_match else None
                
                result_ready = False
                if tid and target_pl:
                    search_pattern = f"{self.write_results_root}/resultsGym_*/{self.test_dataset}/LTF_{tid}_*_pl{target_pl}_*"
                    res = glob.glob(search_pattern)
                    if res:
                        res.sort(key=os.path.getmtime, reverse=True)
                        best_folder = res[0]
                        metric_file = os.path.join(best_folder, 'metrics.npy')
                        pred_file = os.path.join(best_folder, 'pred.npy')
                        true_file = os.path.join(best_folder, 'true.npy')
                        if os.path.exists(metric_file) and os.path.exists(pred_file) and os.path.exists(true_file):
                            result_ready = True
                            logger.info(f"[expand_testset] Results already exist for {best_name}")
                            try:
                                metrics = np.load(metric_file)
                                selected_script_metrics['mae'] = float(metrics[0])
                                selected_script_metrics['mse'] = float(metrics[1])
                                selected_script_metrics['result_folder'] = best_folder
                            except Exception as e:
                                logger.error(f"[expand_testset] Error loading existing metrics: {e}")
                                result_ready = False
                
                if not result_ready:
                    logger.info(f"[expand_testset] Results missing for {best_name}, executing script...")
                    try:
                        import subprocess, tempfile
                        run_script_path = os.path.abspath(self.meta_run_script)
                        with open(script_path, 'r') as f:
                            lines = f.readlines()
                        cmd = ' '.join(ln.strip().rstrip('\\') for ln in lines if ln.strip() != '')
                        cmd = cmd.replace('run.py', run_script_path)
                        cmd = cmd.rstrip() + "\n"
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sh')
                        tmp.write(cmd.encode('utf-8'))
                        tmp.close()
                        env = os.environ.copy()
                        env['RESULTS_ROOT'] = self.write_results_root
                        if self.cuda_devices:
                            env['CUDA_VISIBLE_DEVICES'] = str(self.cuda_devices[0])
                        subprocess.check_call(['bash', tmp.name], env=env, cwd=None)
                        os.unlink(tmp.name)
                        
                        if tid and target_pl:
                            search_pattern = f"{self.write_results_root}/resultsGym_*/{self.test_dataset}/LTF_{tid}_*_pl{target_pl}_*"
                            res = glob.glob(search_pattern)
                            if res:
                                res.sort(key=os.path.getmtime, reverse=True)
                                best_folder = res[0]
                                metric_file = os.path.join(best_folder, 'metrics.npy')
                                if os.path.exists(metric_file):
                                    metrics = np.load(metric_file)
                                    selected_script_metrics['mae'] = float(metrics[0])
                                    selected_script_metrics['mse'] = float(metrics[1])
                                    selected_script_metrics['result_folder'] = best_folder
                    except Exception as e:
                        logger.error(f"[expand_testset] failed to run best script: {e}")
        else:
            if self.ensemble_enabled:
                # 非expand模式：集成预测的top5
                ens_mae, ens_mse, individual_metrics = self.ensemble_predictions(
                    topk_names, self.test_dataset, root_path=self.write_results_root
                )
            else:
                logger.info('Ensemble disabled; skipping ensemble_predictions.')
        
        # =================== 保存结果 ===================
        cross_dataset_perf = self.get_cross_dataset_performance(topk_names[0]) if not self.expand_testset else {}
        
        # 构建文件名
        max_size_suffix = f'-max_size_{self.max_size}' if self.max_size is not None else ''
        expand_suffix = f'-expand_testset' if self.expand_testset else ''
        clip_suffix = f'-clip_{self.clip_timestamps}{int(self.cutoff_time)}' if self.clip_timestamps else ''
        
        # 添加超参数到文件名
        k_val = getattr(self, "k", 0.0)
        temp_val = getattr(self, "temporal", 1.0)
        if "icl" in getattr(self, "meta_model_type", ""):
            icl_suffix = f'-ishuf_{self.icl_shuffle}-ibatch_{self.icl_batch}'
            ktemp_suffix = f"-k_{k_val}-temp_{temp_val}"
        else:
            icl_suffix = ''
            ktemp_suffix = ""
        hp_suffix = f'-lr_{self.lr}-dm_{self.d_model}-nl_{self.n_layers}-wd_{self.weight_decay}{icl_suffix}{ktemp_suffix}'
        
        # =================== 保存最佳checkpoint ===================
        checkpoint_base_path = (
            f'./meta/checkpoints/checkpoints_{self.meta_feature_type}/'
            f'{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}{hp_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_legacy'
        )
        
        # 保存最佳模型checkpoint
        self.save_checkpoint(
            model_state=best_model_wts,
            save_path=f"{checkpoint_base_path}_best",
            best_epoch=best_epoch,
            val_loss=best_metric,
            fold_name='legacy_70_30_split'
        )
        logger.info(f"Best model checkpoint saved (epoch={best_epoch}, val_loss={best_metric:.6f})")
        
        save_path = (
            f'./meta/results/results_{self.meta_feature_type}/'
            f'{self.suffix + "_" if self.suffix else ""}{self.test_dataset}-model_{self.meta_model_type}'
            f'-component_balance_{self.arg_component_balance}'
            f'-add_transformer_{self.arg_add_transformer}'
            f'-add_LLM_{self.arg_add_LLM}'
            f'-add_TSFM_{self.arg_add_TSFM}'
            f'-add_GRU_{self.arg_add_GRU}'
            f'-all_periods_{self.arg_all_periods}'
            f'-ensemble_{self.ensemble_enabled}'
            f'{max_size_suffix}{expand_suffix}{clip_suffix}{hp_suffix}'
            f'_{self.pred_len_1}_{self.pred_len_2}_legacy.npz'
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez_compressed(
            save_path,
            # 超参数
            hyperparams=self.hyperparams,
            
            # 单模型结果（无K折）
            k_folds=1,
            training_mode='legacy_70_30_split',
            best_epoch=best_epoch,
            best_val_loss=best_metric,
            
            # 预测结果
            predictions=y_preds,
            pred_ranks_for_true_topk=pred_ranks_for_true_topk,
            true_ranks_for_pred_topk=true_ranks_for_pred_topk,
            total_num=len(y_preds),
            top1_perf=top1_perf,
            top1_perf_mae=top1_perf_mae,
            top_name=top_name,
            topk_names=topk_names,
            
            # 集成组合的结果
            ensemble_metrics_best=[ens_mae, ens_mse],
            individual_model_metrics=individual_metrics,
            
            # 其他
            cross_dataset_performance=cross_dataset_perf,
            expanded_testset_selected_metrics=selected_script_metrics,
            test_dataset=self.test_dataset,
            pred_len_1=self.pred_len_1,
            pred_len_2=self.pred_len_2,
            epoch_metrics=epoch_metrics_history,
        )
        
        logger.info(f"Results saved to: {save_path}")
        
        # =================== 保存 Attention Weights（如果启用，Legacy 模式）===================
        if should_save_attention and attention_info is not None:
            # 构建 legacy 模式的 attention 数据
            legacy_attn = {
                'N_train': attention_info['N_train'],
                'N_test': attention_info['N_test'],
            }
            for key in ['train_to_train', 'train_to_test', 'test_to_train', 'test_to_test']:
                if key in attention_info and attention_info[key] is not None:
                    legacy_attn[key] = attention_info[key]
            
            if any(k in legacy_attn for k in ['train_to_train', 'train_to_test', 'test_to_train', 'test_to_test']):
                attention_save_path = save_path.replace('.npz', '_attention.npz')
                np.savez_compressed(
                    attention_save_path,
                    # 基本信息
                    test_dataset=self.test_dataset,
                    pred_len_1=self.pred_len_1,
                    pred_len_2=self.pred_len_2,
                    meta_model_type=self.meta_model_type,
                    k_folds=1,
                    training_mode='legacy_70_30_split',
                    
                    # Attention 数据（legacy 模式只有一个模型）
                    # 格式与 kfold 保持一致，使用 attention_data 字典
                    # attention_data['legacy'] = {
                    #   'N_train', 'N_test',
                    #   'train_to_train': (num_layers, N_train, N_train) - TL block,
                    #   'train_to_test': (num_layers, N_train, N_test) - TR block (通常被 mask),
                    #   'test_to_train': (num_layers, N_test, N_train) - BL block,
                    #   'test_to_test': (num_layers, N_test, N_test) - BR block,
                    # }
                    attention_data={'legacy': legacy_attn},
                    
                    # 测试集组件名称（用于分析）
                    test_component_names=np.array(self.name_components, dtype=object),
                )
                logger.info(f"Attention weights saved to: {attention_save_path}")
        
        return {
            'predictions': y_preds,
            'top_name': top_name,
            'topk_names': topk_names,
            'top1_perf': top1_perf,
            'best_epoch': best_epoch,
            'epoch_metrics': epoch_metrics_history,
        }


if __name__ == "__main__":
    # =================== 参数解析 ===================
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_feature_type', type=str, default='tabpfn_samplesdynamic_windowlen50_seed42')
    parser.add_argument('--clip_timestamps', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--read_results_root', type=str, default='./results_long_term_forecasting',
                        help='where to read existing large-scale experiment results')
    parser.add_argument('--write_results_root', type=str, default='./meta/results_long_term_forecasting',
                        help='where to write any rerun (ensemble) outputs to avoid polluting originals')
    parser.add_argument('--enable_ensemble', type=lambda x: str(x).lower() == 'true', default=False,
                        help='whether to run ensemble_predictions at the end')
    parser.add_argument('--meta_run_script', type=str, default='run_meta_forecast.py',
                        help='runner script used when re-launching missing experiments')
    parser.add_argument('--cuda_devices', type=str, default='0,1,2,3,4,5,6',
                        help='comma-separated CUDA device IDs to use for parallel reruns')
    parser.add_argument('--parallel_workers', type=int, default=10,
                        help='max parallel workers; default uses number of CUDA devices')
    parser.add_argument('--meta_model_type', type=str, default='mlp', 
                        choices=['mlp', 'icl-nomasktrain', 'icl-mq', 'icl-yx', 'icl-ls', 'icl-hls', 'icl-nomask', 
                                'icl-simple', 'icl-simple-nomask', 'icl-simplemask', 'baseline-nn',
                                'baseline-nn-datasetsensemble', 'baseline-nn-componentsensemble',
                                'icl-frozencomp', 'icl-addcomp', 'icl-labelencoder', 'icl-nomasktrain-deepinput',
                                'fewshot-best', 'icl-tabpfn'],
                        help='Meta model type: mlp, icl variants (with Q/K/V), icl-simple variants (no Q/K/V projection), baseline-nn variants, or new variants (frozencomp, addcomp, labelencoder, deepinput, icl-tabpfn)')
    parser.add_argument('--max_size', type=int, default=None,
                        help='Maximum number of results to use, limited by TSGym ID order.')
    parser.add_argument('--expand_testset', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, expand testset_components by loading scripts from meta_script_root')
    parser.add_argument('--meta_script_root', type=str, default='./meta/script',
                        help='Root directory for expanded testset scripts.')

    # 新增超参数
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for meta-learner')
    parser.add_argument('--d_model', type=int, default=64, help='Embedding dimension for meta-learner')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in meta-learner (MLP和ICL共用)')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads for MetaICL')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for meta-learner')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--es_tol', type=int, default=5, help='Early stopping tolerance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--rank_aggregation_method', type=str, default='median',
                        choices=['mean', 'median', 'trimmed_mean', 'voting'],
                        help='Rank aggregation method: mean, median (robust), trimmed_mean, voting')

    # ICL-specific hyperparameters
    parser.add_argument('--k', type=float, default=0.0,
                        help='Percentage k (0-100) for truncating smallest attention scores in ICL attention (before softmax).')
    parser.add_argument('--temporal', type=float, default=1.0,
                        help='Softmax temperature for attention in ICL-based meta-learner (T=1.0 means no scaling).')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix to prepend to the result identifier (e.g. for marking experiments).')



    # 新增：并行 K 折训练支持
    parser.add_argument('--fold_idx', type=int, default=None,
                        help='Run only the specified fold index (0-based). If not set, run all folds sequentially. '
                            'Use this to run different folds in parallel on different GPUs/terminals.')
    parser.add_argument('--ensemble_only', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, only ensemble from saved fold results without training. '
                            'Use this after all parallel fold trainings are complete.')

    # 新增：ICL 变体控制
    parser.add_argument('--icl_shuffle', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, shuffle training data before each epoch (for ICL full-batch training).')
    parser.add_argument('--icl_batch', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, use mini-batch training for ICL models instead of full-batch.')
    parser.add_argument('--save_attention', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, save attention weights from ICL models during testing for later analysis. '
                            'Attention weights will be saved to a separate *_attention.npz file.')

    # 新增：组件控制参数
    parser.add_argument('--arg_component_balance', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, balance the number of samples across datasets by random sampling.')
    parser.add_argument('--arg_add_GRU', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, include GRU model results in the training/testing pool.')
    parser.add_argument('--arg_add_transformer', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, include Transformer model results in the training/testing pool.')
    parser.add_argument('--arg_add_LLM', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, include LLM model results in the training/testing pool.')
    parser.add_argument('--arg_add_TSFM', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, include TSFM model results in the training/testing pool.')
    parser.add_argument('--arg_all_periods', type=lambda x: str(x).lower() == 'true', default=False,
                        help='If true, use all pred_lens (all periods) in training data instead of single pred_len.')

    args = parser.parse_args()

    meta_feature_type = args.meta_feature_type
    clip_timestamps = args.clip_timestamps
    # line 139 to change timestamps
    read_results_root = args.read_results_root
    write_results_root = args.write_results_root
    enable_ensemble = args.enable_ensemble
    meta_run_script = args.meta_run_script

    os.makedirs('meta/logfiles', exist_ok=True)
    os.makedirs(f'meta/results/results_{meta_feature_type}', exist_ok=True)
    os.makedirs(f'meta/checkpoints/checkpoints_{meta_feature_type}', exist_ok=True)
    
    # New plot directory
    plot_dir_base = os.path.join(f'meta/result_plots', meta_feature_type)
    os.makedirs(plot_dir_base, exist_ok=True)

    # 设置全局随机种子
    set_seed(args.seed)

    # 构建实验设置标识符
    fold_suffix = f"-fold_{args.fold_idx}" if args.fold_idx is not None else ""
    ensemble_only_suffix = "-ensemble_only" if args.ensemble_only else ""
    exp_setting_id = (
    f"model_{args.meta_model_type}"
    f"-feat_{meta_feature_type}"
    f"-clip_{clip_timestamps}"
    f"-maxsize_{args.max_size}"
    f"-expand_{args.expand_testset}"
    f"-ensemble_{enable_ensemble}"
    f"-lr_{args.lr}"
    f"-dm_{args.d_model}"
    f"-nl_{args.n_layers}"
    f"-nh_{args.nhead}"
    f"-do_{args.dropout}"
    f"-kfold_{args.use_kfold}"
    f"-ishuf_{args.icl_shuffle}"
    f"-ibatch_{args.icl_batch}"
    f"-k_{args.k}"
    f"-temp_{args.temporal}"
    f"-balance_{args.arg_component_balance}"
    f"-gru_{args.arg_add_GRU}"
    f"-tsfmer_{args.arg_add_transformer}"
    f"-llm_{args.arg_add_LLM}"
    f"-tsfm_{args.arg_add_TSFM}"
    f"-allperiods_{args.arg_all_periods}"
    f"{fold_suffix}{ensemble_only_suffix}"
    )
    
    # Simplified ID for Plotting (to avoid path length issues)
    plot_setting_id = (
        f"{args.suffix + '_' if args.suffix else ''}"
        f"model_{args.meta_model_type}"
        f"-nl_{args.n_layers}"
        f"-lr_{args.lr}"
        f"-maxsize_{args.max_size}"
        f"-ibatch_{args.icl_batch}"
        f"{fold_suffix}{ensemble_only_suffix}"
    )
    
    # Ensure plot directory for this setting exists
    # Structure: meta/result_plots/{meta_feature_type}/{exp_setting_id}/testdataset{test_dataset}/predlen{pl}/plot_{dataset}.png
    
    def _plot_epoch_metrics(epoch_metrics, test_dataset, pred_len, best_epoch):
        """Helper to plot epoch metrics"""
        if not epoch_metrics: return

        # Identify all datasets present in metrics
        datasets = set()
        for em in epoch_metrics:
            for k in em.keys():
                if k not in ['epoch', 'val_loss']:
                    datasets.add(k)
        
        # Prepare directory
        # Using simplified plot_setting_id
        save_dir = os.path.join(plot_dir_base, plot_setting_id, f"testdataset{test_dataset}", f"predlen{pred_len}")
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = [em['epoch'] for em in epoch_metrics]
        
        for d_name in datasets:
            ranks = [em.get(d_name, {}).get('rank', np.nan) for em in epoch_metrics]
            mses = [em.get(d_name, {}).get('mse', np.nan) for em in epoch_metrics]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Left Axis: Rank
            color = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Top-1 Rank (Higher Y is Better)', color=color)
            ax1.plot(epochs, ranks, color=color, label='Rank', marker='o', markersize=3)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Make Y-axis ticks more dense and integer-based
            from matplotlib.ticker import MaxNLocator
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20)) 
            ax1.grid(True, which='both', axis='y', linestyle=':', alpha=0.5)
            
            ax1.invert_yaxis() # Rank 1 is top
            
            # Right Axis: MSE
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Top-1 MSE (Lower Y is Better)', color=color)
            ax2.plot(epochs, mses, color=color, label='MSE', linestyle='dashed', marker='x', markersize=3)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Best Epoch Line
            plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
            
            dataset_type = "TEST" if d_name == test_dataset else "TRAIN"
            plt.title(f"{args.meta_model_type}-pl{pred_len}-{d_name}-{dataset_type}")
            fig.tight_layout()
            
            plot_path = os.path.join(save_dir, f"plot_{d_name}.png")
            plt.savefig(plot_path)
            plt.close(fig)
            
        logger.info(f"Performance plots saved to {save_dir}")

    # 配置日志：同时输出到文件和控制台
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的handler（避免重复添加）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 日志格式
    log_format = logging.Formatter('%(asctime)s - %(message)s')

    # 文件handler - 使用简短的文件名（模型类型 + 哈希）避免文件名过长错误
    import hashlib
    exp_setting_hash = hashlib.md5(exp_setting_id.encode()).hexdigest()[:12]
    log_filename = f'meta/logfiles/meta_{args.meta_feature_type}_{args.meta_model_type}_{exp_setting_hash}.log'
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    logger.info(f"Experiment Setting: {exp_setting_id}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"NumPy version: {np.__version__}")

    # 创建Meta对象
    meta = Meta(
        seed=args.seed,
        read_results_root=read_results_root,
        write_results_root=write_results_root,
        ensemble_enabled=enable_ensemble,
        meta_run_script=meta_run_script,
        cuda_devices=args.cuda_devices,
        parallel_workers=args.parallel_workers,
        meta_model_type=args.meta_model_type,
        max_size=args.max_size,
        expand_testset=args.expand_testset,
        meta_script_root=args.meta_script_root,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        es_tol=args.es_tol,
        rank_aggregation_method=args.rank_aggregation_method,
        icl_shuffle=args.icl_shuffle,
        icl_batch=args.icl_batch,
        save_attention=args.save_attention,
        k=args.k,
        temporal=args.temporal,
        suffix=args.suffix,
    )

    task_name = 'LTF'
    # 从命令行参数读取组件控制变量
    if args.arg_add_LLM or args.arg_add_TSFM:
        datasets = ['ETTh1', 'ETTh2', 'Exchange', 'ili']
    else:
        datasets = sorted([_ for _ in os.listdir('./results_long_term_forecasting/resultsGym_MLP')])
    #  不要nyse和nasdaq了
    datasets = [_ for _ in datasets if _ not in ['nyse','nasdaq']]
    if 'part-datasets' in args.suffix:
        datasets = ['ECL','ETTh1','ETTh2','ETTm1','ETTm2','traffic','weather']
    if 'dropECLtraffic' in args.suffix:
        datasets = ['Exchange','ETTh2','ETTm1','ETTm2','weather','ETTh1','ili']

    # =================== 主训练循环 ===================
    def run_components_processing(test_dataset, pred_len_1, pred_len_2):
        """执行数据处理"""
        meta.components_processing(
            task_name=task_name,
            datasets=datasets,
            meta_feature_type=meta_feature_type,
            test_dataset=test_dataset,
            pred_len_1=pred_len_1,
            pred_len_2=pred_len_2,
            arg_component_balance=args.arg_component_balance,
            arg_add_transformer=args.arg_add_transformer,
            arg_add_LLM=args.arg_add_LLM,
            arg_add_TSFM=args.arg_add_TSFM,
            arg_add_GRU=args.arg_add_GRU,
            arg_all_periods=args.arg_all_periods,
            clip_timestamps=clip_timestamps
        )

    def run_training_step():
        """执行训练步骤（不包含数据处理）"""
        results = None
        if args.meta_model_type == 'baseline-nn':
            logger.info("Running Baseline: Nearest Neighbor (no training)")
            results = meta.meta_fit_baseline_nn()
        elif args.meta_model_type == 'baseline-nn-datasetsensemble':
            logger.info("Running Baseline: Nearest Neighbor Dataset Ensemble (top-3 datasets, top-1 combo each)")
            results = meta.meta_fit_baseline_nn_dataset_ensemble(topk_datasets=3)
        elif args.meta_model_type == 'baseline-nn-componentsensemble':
            logger.info("Running Baseline: Nearest Neighbor Components Ensemble (top-1 dataset, top-5 combos)")
            results = meta.meta_fit_baseline_nn_components_ensemble(topk_components=5)
        elif args.meta_model_type == 'fewshot-best':
            logger.info("Running Meta-Learner: Fewshot-Best (Oracle Baseline)")
            results = meta.meta_fit_fewshot_best()

        else:
            meta.meta_init()
            train_res = meta.meta_fit()
            # If standard mode, we should pass this history to evaluation/saving?
            # meta_fit already saves the result for the current pred_len within the method.
            # So we just return the results.
            results = train_res
        return results

    def run_training(test_dataset, pred_len_1, pred_len_2):
        """执行单次完整的数据处理和训练"""
        run_components_processing(test_dataset, pred_len_1, pred_len_2)
        results = run_training_step()
        
        # Plotting
        if results and 'epoch_metrics' in results:
             best_epoch = results.get('best_epoch', 0)
             # Determine pred_len for plotting directory
             # For 'not arg_all_periods', it's usually pred_len_1 unless it's the special datasets
             target_pl = pred_len_2 if test_dataset in ['ili', 'nyse', 'nasdaq'] else pred_len_1
             _plot_epoch_metrics(results['epoch_metrics'], test_dataset, target_pl, best_epoch)
             
        return results

    def run_all_periods_mode(test_dataset):
        """
        arg_all_periods=True 模式：
        训练集包含所有 pred_len，只训练一次，然后对每个 pred_len 的测试集分别评估
        """
        logger.info(f"[arg_all_periods=True] Training once for {test_dataset}, then evaluating on each pred_len")
        
        trained_fold_results = None  # 保存训练好的模型状态 (for kfold mode)
        trained_legacy_model_state = None  # 保存训练好的模型状态 (for legacy mode)
        pred_lens = list(zip([96, 192, 336, 720], [24, 36, 48, 60]))
        
        for idx, (pred_len_1, pred_len_2) in enumerate(pred_lens):
            # 处理数据（更新测试集为当前 pred_len）
            run_components_processing(test_dataset, pred_len_1, pred_len_2)
            
            if idx == 0:
                # 第一个 pred_len：完整训练 + 测试
                logger.info(f"[arg_all_periods=True] Training with pred_len={pred_len_1}/{pred_len_2}")
                results = run_training_step()
                
                elif not args.use_kfold and hasattr(meta, 'model'):
                    # Legacy mode: save the single model's state
                    trained_legacy_model_state = copy.deepcopy(meta.model.state_dict())
                    logger.info(f"[arg_all_periods=True] Model trained (legacy), saved model state")
            else:
                # 后续 pred_len：只使用已训练模型进行评估
                logger.info(f"[arg_all_periods=True] Evaluating with pred_len={pred_len_1}/{pred_len_2} (using trained models)")
                
                if args.meta_model_type == 'baseline-nn':
                    results = meta.meta_fit_baseline_nn()
                elif args.meta_model_type == 'baseline-nn-datasetsensemble':
                    results = meta.meta_fit_baseline_nn_dataset_ensemble(topk_datasets=3)
                elif args.meta_model_type == 'baseline-nn-componentsensemble':
                    results = meta.meta_fit_baseline_nn_components_ensemble(topk_components=5)
#                 elif args.use_kfold and trained_fold_results is not None:
#                     results = meta.evaluate_with_trained_models(trained_fold_results)
                elif not args.use_kfold and trained_legacy_model_state is not None:
                    # Legacy mode: reload the saved model and run prediction
                    meta.meta_init()  # Reinitialize model structure
                    meta.model.load_state_dict(trained_legacy_model_state)
                    # Run prediction on the new test set (meta_predict handles this)
                    results = meta._evaluate_legacy_with_trained_model(trained_legacy_model_state)
                else:
                    logger.warning(f"Skipping evaluation for pred_len={pred_len_1}/{pred_len_2}: no trained models available")

    # =================== 执行主循环 ===================

    print(datasets)
    for test_dataset in datasets:

        if args.arg_all_periods:
            # 优化模式：训练一次，评估四次
            run_all_periods_mode(test_dataset)
        else:
            # 标准模式：每个 pred_len 分别训练
            for pred_len_1, pred_len_2 in zip([96, 192, 336, 720], [24, 36, 48, 60]):

                logger.info(f"[Standard mode] Training for {test_dataset} with pred_len_1={pred_len_1}, pred_len_2={pred_len_2}")
                run_training(test_dataset, pred_len_1, pred_len_2)
