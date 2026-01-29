# 三种meta-feature方法：TSFEL，TSFused，Foundation models
import tsfel
import numpy as np
import pandas as pd
import warnings
import tsfel
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf, adfuller
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_provider.m4 import M4Dataset, M4Meta
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
print("TabPFN Extensions imported successfully.")
import warnings
warnings.filterwarnings("ignore")
# os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"



def read_data(file_path):
    """
    read data from file_path, return training set
    Logic aligned with data_provider/data_loader.py
    """
    # check
    if not os.path.isfile(file_path):
        assert FileNotFoundError
    df_raw = pd.read_csv(file_path, header=0)
    df_raw = df_raw.dropna(axis=1, how='all')  # in case there is a column with all nans
    df_raw = df_raw.drop(columns=['date','Date','timestamp'], errors='ignore')
    
    filename = os.path.basename(file_path)
    
    if filename in ['ETTh1.csv', 'ETTh2.csv']:
        # Dataset_ETT_hour
        border1 = 0
        border2 = 12 * 30 * 24
        data = df_raw.iloc[border1:border2]
    elif filename in ['ETTm1.csv', 'ETTm2.csv']:
         # Dataset_ETT_minute
        border1 = 0
        border2 = 12 * 30 * 24 * 4
        data = df_raw.iloc[border1:border2]
    else:
        # Dataset_Custom
        # num_train = int(len(df_raw) * 0.7)
        # Using 0.7 for training as in Dataset_Custom default
        train_ratio = 0.7
        num_train = int(len(df_raw) * train_ratio)
        border1 = 0
        border2 = num_train
        data = df_raw.iloc[border1:border2]

    return data


# 基于TSFEL提取meta-features
def get_meta_feature_tsfel(file_path=None, data=None, target_dim=20000, seed=42):
    if data is None:
        data = read_data(file_path)
        data = data.values.astype(np.float32) # tsfel要求输入为numpy array
    else:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = data.astype(np.float32)
    
    # get meat feature
    # Extracts the temporal, statistical and spectral feature sets.
    # Returns a DataFrame with the features.
    cfg = tsfel.get_features_by_domain() 
    
    meta_feature_list = []
    # Iterate over each variable (column) in the time series
    for i in range(data.shape[1]):
        # Extract features for the single time series
        # Returns shape (1, n_features)
        X = tsfel.time_series_features_extractor(cfg, data[:,i], fs=100, verbose=0).values
        meta_feature_list.append(X)
    
    # Concatenate to shape (n_variables, n_features)
    meta_feature = np.concatenate(meta_feature_list, axis=0)
    
    # Aggregate over variables using 9 statistics to capture the distribution of features across the variables
    # This results in shape (9, n_features)
    mean = np.mean(meta_feature, axis=0)
    std = np.std(meta_feature, axis=0)
    min_val = np.min(meta_feature, axis=0)
    q25 = np.percentile(meta_feature, 25, axis=0)
    median = np.median(meta_feature, axis=0)
    q75 = np.percentile(meta_feature, 75, axis=0)
    max_val = np.max(meta_feature, axis=0)
    range_val = max_val - min_val
    iqr = q75 - q25
    
    combined_features = np.stack([mean, std, min_val, q25, median, q75, max_val, range_val, iqr])
    # combined_features shape is approx (9, 156) ~ 1404 dimensions flattened
    
    # Flatten high-dimensional feature matrix
    features_flat = combined_features.flatten().reshape(1, -1)
    # fillna 0
    features_flat = np.nan_to_num(features_flat, nan=0.0)
    
    # Reduce dimensionality to target_dim (200) using Gaussian Random Projection
    # A fixed random_state ensures the projection matrix is the same for every dataset,
    # satisfying the requirement of consistency without using other datasets' data.
    if features_flat.shape[1] > target_dim:
        transformer = GaussianRandomProjection(n_components=target_dim, random_state=seed)
        features_reduced = transformer.fit_transform(features_flat)
        return features_reduced.flatten()
    else:
        return features_flat.flatten()


# 基于TSFused提取meta-features
def get_meta_feature_tsfused(file_path=None, data=None):
    """
    Extracts meta-features from a given time series data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features), time series data

    Returns:
    - features: dict, contains the extracted meta-features
    """
    if data is None:
        data = read_data(file_path)
        data = data.values.astype(np.float32) # tsfel要求输入为numpy array
    else:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = data.astype(np.float32)
    features = {}

    # basic statistics
    features["mean"] = np.mean(data, axis=0).mean()
    features["std"] = np.std(data, axis=0).mean()
    features["min"] = np.min(data, axis=0).mean()
    features["max"] = np.max(data, axis=0).mean()
    features["skewness"] = np.nanmean(skew(data, axis=0))
    features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))

    # time series decomposition
    acfs = [acf(data[:, i], nlags=10, fft=True) for i in range(data.shape[1])]
    features["autocorrelation_mean"] = np.nanmean(
        [acf_val[1] for acf_val in acfs]
    )  # first lag
    adf_results = []
    for i in range(data.shape[1]):
        try:
            adf_results.append(adfuller(data[:, i]))
        except:
            adf_results.append((np.nan, 0.0))  # If ADF fails, assume non-stationary
    features["stationarity"] = np.mean([result[1] < 0.05 for result in adf_results])

    # rate_of_change = np.diff(data, axis=0) / data[:-1]
    # Deal with 0 division
    safe_data = np.where(data[:-1] == 0, np.nan, data[:-1])
    rate_of_change = np.diff(data, axis=0) / safe_data
    features["rate_of_change_mean"] = np.nanmean(rate_of_change)
    features["rate_of_change_std"] = np.nanstd(rate_of_change)

    # Landmarker features
    autoreg_coefs, residual_stds = [], []
    for i in range(data.shape[1]):
        model = AutoReg(data[:, i], lags=1).fit()
        autoreg_coefs.append(model.params[1])
        residual_stds.append(np.std(model.resid))
    features["autoreg_coef_mean"] = np.mean(autoreg_coefs)
    features["residual_std_mean"] = np.mean(residual_stds)

    # frequency domain features
    freq_means, freq_peaks, spectral_entropies = [], [], []
    spectral_variations, spectral_skewnesses, spectral_kurtoses = [], [], []

    for i in range(data.shape[1]):
        freqs, psd = periodogram(data[:, i])
        freq_means.append(np.mean(psd))
        freq_peaks.append(freqs[np.argmax(psd)])
        spectral_entropies.append(entropy(psd))
        if i > 0:
            prev_psd = periodogram(data[:, i - 1])[1]
            spectral_variations.append(np.sqrt(np.sum((psd - prev_psd) ** 2)))
        else:
            spectral_variations.append(0)  # 第一个变量无法计算变化
        spectral_skewnesses.append(skew(psd))
        spectral_kurtoses.append(kurtosis(psd))

    features["frequency_mean"] = np.mean(freq_means)
    features["frequency_peak"] = np.mean(freq_peaks)
    features["spectral_entropy"] = np.nanmean(spectral_entropies)
    features["spectral_variation"] = np.nanmean(spectral_variations)
    features["spectral_skewness"] = np.nanmean(spectral_skewnesses)
    features["spectral_kurtosis"] = np.nanmean(spectral_kurtoses)

    cov_matrix = np.cov(data, rowvar=False)
    features["covariance_mean"] = np.mean(cov_matrix)
    features["covariance_max"] = np.max(cov_matrix)
    features["covariance_min"] = np.min(cov_matrix)
    features["covariance_std"] = np.std(cov_matrix)
    # dict to numpy array
    features = np.array(list(features.values()))
    return features


# 构造自监督任务 (Self-Supervised Task) 用于 TabPFN
# 由于 TabPFN 是分类模型，我们可以通过"预测下一个时间步的值（离散化后）"来构造标签
def prepare_tabpfn_data(file_path=None, data=None, window_len=50, n_samples=None, n_classes=10, seed=42, samples_per_feature=100):
    """
    构造 (N, T) 的样本和 (N,) 的分类标签
    
    Parameters:
    - samples_per_feature: 每个特征平均采样的次数，用于确保多变量数据集被充分采样
    """
    if seed is not None:
        np.random.seed(seed)
        
    if data is None:
        df = read_data(file_path) # 使用前面定义的 read_data
        data = df.values
    else:
        if data.ndim == 1:
            data = data.reshape(-1, 1)

    n_timesteps, n_features = data.shape
    
    # Adjust window_len if data is too short
    if n_timesteps <= window_len:
        # User requested: "if a series is too short, windowlen cannot exceed its own length"
        # We need at least 1 point for target, so max window size is n_timesteps - 1
        effective_window_len = n_timesteps - 1
    else:
        effective_window_len = window_len
    print(f"Effective window length: {effective_window_len}")
        
    if effective_window_len < 1:
        # Too short to do anything
        return np.zeros((0, window_len)), np.zeros((0,))

    # Limit n_samples if data is short
    # If we have very few possible starts, we can't sample more than that effectively without massive duplication
    possible_starts = n_timesteps - effective_window_len # e.g. 10 - 9 = 1 start at index 0
    if possible_starts <= 0:
        return np.zeros((0, effective_window_len)), np.zeros((0,))
        
    # If n_samples is None, use logic, otherwise use provided n_samples
    if n_samples is None:
        n_samples = max(2000, n_timesteps // 2, n_features * samples_per_feature)
        n_samples = min(n_samples, 50000)
    
    # Check if we are sampling more than possible unique windows? 
    # For M4 univariate, if possible_starts is small (e.g. 5), asking for 2000 samples is redundant.
    # But TabPFN might benefit from "bootstrapping"? Or maybe not.
    # Let's cap n_samples for very short series to avoid unnecessary compute?
    # Actually, let's keep n_samples logic but just random sample with replacement as implemented below.

    # name = file_path if file_path else "InMemoryData"
    # print(f"Preparing TabPFN data for {name}: window_len={effective_window_len}, n_samples={n_samples}, n_classes={n_classes}, n_features={n_features}")
    
    # 简单标准化，防止某些序列数值过大
    # Handle std=0 case
    std_val = np.nanstd(data, axis=0)
    data = (data - np.nanmean(data, axis=0)) / (std_val + 1e-8)
    
    X_list = []
    y_list = []
    
    for _ in range(n_samples):
        # 随机选一个变量
        feat_idx = np.random.randint(0, n_features)
        # 随机选一个起始点 [0, possible_starts)
        start_idx = np.random.randint(0, possible_starts)
        
        window = data[start_idx : start_idx + effective_window_len, feat_idx]
        target = data[start_idx + effective_window_len, feat_idx]
        
        X_list.append(window)
        y_list.append(target)
        
    X = np.stack(X_list)
    y_continuous = np.array(y_list)
    
    # 将连续目标离散化为类别 (Binning)
    # 使用分位数分桶，保证类别平衡
    # Check uniqueness
    if len(np.unique(y_continuous)) < n_classes:
         # If fewer unique values than classes, just map to ranks or reduce classes?
         # Simplest is pd.cut with ranks or just whatever
         y = pd.cut(y_continuous, bins=min(n_classes, len(np.unique(y_continuous))), labels=False)
    else:
        try:
            y = pd.qcut(y_continuous, q=n_classes, labels=False, duplicates='drop')
        except ValueError:
            # 如果数据过于集中导致分位数重复，使用等宽分桶
            y = pd.cut(y_continuous, bins=n_classes, labels=False)
    
    # 处理可能的 NaN (pd.cut 可能会产生 NaN)
    y = np.nan_to_num(y, nan=0).astype(int)
        
    return X, y

def get_tabpfn_embedding(file_path=None, data=None, window_len=50, n_samples=None, n_classes=10, seed=42, samples_per_feature=100, embedding_extractor=None):
    """
    使用 TabPFN 提取时间序列的 Embedding
    """
    X, y = prepare_tabpfn_data(file_path, data, window_len, n_samples, n_classes, seed, samples_per_feature)
    if X.shape[0] == 0:
        # 如果没有样本，返回全零向量
        return np.zeros((256,)) # TabPFN embedding dim is usually larger? Check default config, 512?
        # Actually default TabPFN classifier small has 512, big might have more?
        # But if we pass it later, it will likely mismatch if wrong. 
        # Better to return None or check extractor. 
        # For now assume failure/empty returns zero of reasonable size or handle upstream.
        # But let's look at embedding_extractor if available.
        
    
    # 初始化 TabPFN Embedding 模型 (if not provided)
    if embedding_extractor is None:
        model_path = "/data2/coding/tsgym/tabfpn-v2/tabpfn-v2.5-classifier-v2.5_default.ckpt"
        classifier = TabPFNClassifier(device="cuda", n_estimators=4, model_path=model_path)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=classifier, n_fold=5)

    
    # 获取 Embedding
    # Note: get_embeddings expects (X, y, X_test, ...) 
    # We use same X for extracting embeddings as training set proxy.
    try:
        train_embeddings_full = embedding_extractor.get_embeddings(X, y, X, data_source="train")
        X_train_emb = train_embeddings_full.mean(axis=0)
        dataset_meta_feature = X_train_emb.mean(axis=0)
        return dataset_meta_feature
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return np.zeros((512,)) # Fallback


def get_basic_embedding(file_path=None, data=None):
    if data is None:
        data = read_data(file_path)
        data = data.values.astype(np.float32) # tsfel要求输入为numpy array
    else:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = data.astype(np.float32)
    features = {}
    features['n_samples'] = data.shape[0]
    features['n_features'] = data.shape[1]
    # basic statistics
    features["mean"] = np.mean(data, axis=0).mean()
    features["std"] = np.std(data, axis=0).mean()
    features["min"] = np.min(data, axis=0).mean()
    features["max"] = np.max(data, axis=0).mean()
    features["skewness"] = np.nanmean(skew(data, axis=0))
    features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))
    features = np.array(list(features.values()))
    return features


def prepare_tabpfn_data_m4(series_list, window_len, n_classes=10):
    """
    Prepare data for TabPFN from a list of time series for a specific frequency.
    
    Parameters:
    - series_list: list of 1D numpy arrays (time series)
    - window_len: int, target window length
    - n_classes: int, number of classes for binning
    
    Returns:
    - X: (N, window_len)
    - y: (N,)
    """
    X_list = []
    y_list = []
    
    for series in series_list:
        # We assume series are training data. 
        # We take the last point as target y, and the preceding window_len points as X.
        
        # Check length
        # We need at least 1 point for y.
        if len(series) < 1:
            continue
            
        y_val = series[-1]
        
        # Context
        if len(series) > 1:
            context = series[:-1]
        else:
            context = np.array([])
            
        # Normalize context
        if len(context) > 0:
            mean = np.mean(context)
            std = np.std(context)
            if std < 1e-8:
                std = 1.0
            context = (context - mean) / std
            
        # Handle Window Length
        if len(context) >= window_len:
            # Take last window_len
            x_window = context[-window_len:]
        else:
            # Pad with zeros at the front
            padding = np.zeros(window_len - len(context))
            if len(context) > 0:
                x_window = np.concatenate([padding, context])
            else:
                x_window = padding
        
        X_list.append(x_window)
        y_list.append(y_val)
        
    if not X_list:
        return np.zeros((0, window_len)), np.zeros((0,))

    X = np.stack(X_list)
    y_continuous = np.array(y_list)
    
    # Binning y to classes
    # Check uniqueness
    n_unique = len(np.unique(y_continuous))
    if n_unique < n_classes:
         y = pd.cut(y_continuous, bins=min(n_classes, max(1, n_unique)), labels=False)
    else:
        try:
            y = pd.qcut(y_continuous, q=n_classes, labels=False, duplicates='drop')
        except ValueError:
            y = pd.cut(y_continuous, bins=n_classes, labels=False)
            
    y = np.nan_to_num(y, nan=0).astype(int)
    
    return X, y

def get_tabpfn_embedding_m4(series_list, window_len, embedding_extractor, seed=42):
    """
    Extract TabPFN embeddings for a list of series WITHOUT batching (all at once).
    Aggregates to a single vector (mean).
    """
    X, y = prepare_tabpfn_data_m4(series_list, window_len)
    
    if len(X) == 0:
        return np.zeros((256,)) # Fallback
    
    n_samples = len(X)
    print(f"Preparing to extract embeddings for {n_samples} samples...")
        
    try:
        # Using X as both train and 'test' (source) to get embeddings for these samples.
        # data_source="train" means we are extracting embeddings from the training set provided.
        embeddings = embedding_extractor.get_embeddings(X, y, X, data_source="train")
        
        # embeddings shape typically: (n_estimators, n_samples, embedding_dim)
        # Average over estimators
        embeddings_avg = embeddings.mean(axis=0) # (n_samples, dim)
        
        # Aggregate to single meta-feature vector
        meta_feature = embeddings_avg.mean(axis=0)
        return meta_feature
        
    except Exception as e:
        print(f"Error in TabPFN embedding extraction: {e}")
        return np.zeros((256,)) # Fallback



def get_meta_faetures(file_paths=None, meta_feature_type='tsfel', save_path=None, n_samples=None, seed=42, window_len=50, samples_per_feature=100):
    print(f"Generating meta-features with type: {meta_feature_type}, n_samples: {n_samples}, seed: {seed}, window_len: {window_len}, samples_per_feature: {samples_per_feature}")
    
    # Set global seed
    if seed is not None:
        np.random.seed(seed)

    if meta_feature_type == 'tsfel':
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfel(_, seed=seed) for _ in tqdm(file_paths)}
        if save_path is None:
            save_path = f"./meta_feature_dict_tsfel_seed{seed}.npz"
        # save_path = save_path if save_path is not None else "./meta_feature_dict_tsfel.npz"
    elif meta_feature_type == 'tsfel_gaussianRandomProjection':
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfel(_, target_dim=256, seed=seed) for _ in tqdm(file_paths)}
        if save_path is None:
            save_path = f"./meta_feature_dict_tsfelGRP_seed{seed}.npz"
        # save_path = save_path if save_path is not None else "./meta_feature_dict_tsfelGRP.npz"
    elif meta_feature_type == 'tsfused':
        # tsfused seems not using random seed heavily, but setting global seed above should help if any
        meta_features = {_.split('/')[-1].split('.')[0]: get_meta_feature_tsfused(_) for _ in tqdm(file_paths)}
        if save_path is None:
            save_path = f"./meta_feature_dict_tsfused_seed{seed}.npz"
        # save_path = save_path if save_path is not None else "./meta_feature_dict_tsfused.npz"
    elif meta_feature_type == 'tabpfn':
        meta_features = {_.split('/')[-1].split('.')[0]: get_tabpfn_embedding(_, n_samples=n_samples, seed=seed, window_len=window_len, samples_per_feature=samples_per_feature) for _ in tqdm(file_paths)}
        if save_path is None:
            samples_str = "dynamic" if n_samples is None else n_samples
            save_path = f"./meta_feature_dict_tabpfn_samples{samples_str}_windowlen{window_len}_spf{samples_per_feature}_seed{seed}.npz"
    elif meta_feature_type == 'basic':
        meta_features = {_.split('/')[-1].split('.')[0]: get_basic_embedding(_) for _ in tqdm(file_paths)}
        if save_path is None:
            save_path = f"./meta_feature_dict_basic_seed{seed}.npz"
    elif meta_feature_type == 'tabpfn_basic':
        # reading both tabpfn and basic features and concatenate
        meta_features = {}
        samples_str = "dynamic" if n_samples is None else n_samples
        meta_feautures_tabpfn_path = f"./meta_feature_dict_tabpfn_samples{samples_str}_windowlen{window_len}_spf{samples_per_feature}_seed{seed}.npz"
        meta_features_basic_path = f"./meta_feature_dict_basic_seed{seed}.npz"
        tabpfn_data = np.load(meta_feautures_tabpfn_path, allow_pickle=True)
        basic_data = np.load(meta_features_basic_path, allow_pickle=True)
        for key in tabpfn_data.files:
            tabpfn_feat = tabpfn_data[key]
            basic_feat = basic_data[key]
            combined_feat = np.concatenate([tabpfn_feat, basic_feat])
            meta_features[key] = combined_feat
        if save_path is None:
            save_path = f"./meta_feature_dict_tabpfn_basic_samples{samples_str}_windowlen{window_len}_spf{samples_per_feature}_seed{seed}.npz"
    else:
        raise ValueError("Unknown meta_feature_type: {}".format(meta_feature_type))
    
    print(f"Saving to {save_path}...")
    np.savez(save_path, **meta_features)
    return meta_features

def get_meta_features_m4(root_path, meta_feature_type='tabpfn', save_path=None, seed=42, n_samples=None, **kwargs):
    print(f"Generating M4 meta-features type: {meta_feature_type}...")
    
    if seed is not None:
        np.random.seed(seed)
        
    dataset = M4Dataset.load(training=True, dataset_file=root_path)
    meta_features = {}
    
    embedding_extractor = None
    if 'tabpfn' in meta_feature_type:
        print("Initializing TabPFN model once for the session...")
        model_path = "/data2/coding/tsgym/tabfpn-v2/tabpfn-v2.5-classifier-v2.5_default.ckpt"
        classifier = TabPFNClassifier(device="cuda", n_estimators=4, model_path=model_path)
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=classifier, n_fold=5)

    for sp in M4Meta.seasonal_patterns:
        # Identify Window Length for this Frequency
        # M4Meta.horizons_map gives prediction horizon. 
        # User said: 'Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly' correspond to 6, 8, 18, 13, 14, 48.
        # This matches M4Meta.horizons_map values essentially.
        window_len = M4Meta.horizons_map[sp]
        print(f"Processing M4 {sp}: {len(dataset.groups[dataset.groups == sp])} series. Window length: {window_len}")
        
        # Subset data for this frequency
        indices = np.where(dataset.groups == sp)[0]
        # ids = dataset.ids[indices]
        values = dataset.values[indices]
        
        # Collect valid series (remove NaNs if any, though M4 load handles it, but values[i] might be float array with NaNs padding?
        # M4Dataset.values is object array of numpy arrays, usually checked in load.
        # Actually in data_loader they do: v[~np.isnan(v)]
        
        series_list = []
        for i in range(len(values)):
             v = values[i]
             series_list.append(v[~np.isnan(v)])
             
        print(f"Processing M4 {sp}: {len(series_list)} series. Window length: {window_len}")
        
        if meta_feature_type == 'tabpfn':
             feat = get_tabpfn_embedding_m4(series_list, window_len, embedding_extractor, seed=seed)
             meta_features[sp] = feat
        else:
             print(f"Skipping {sp} for unsupported type {meta_feature_type} in this M4-specific mode")

    if save_path is None:
         save_path = f"./meta_feature_dict_m4_{meta_feature_type}_freq_seed{seed}.npz"
    
    print(f"Saving to {save_path}...")
    np.savez(save_path, **meta_features)
    return meta_features
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Get Meta Features')
    parser.add_argument('--dataset', type=str, default='custom', choices=['custom', 'm4'], help='dataset type')
    parser.add_argument('--root_path', type=str, default="../../dataset", help='root path of the data')
    parser.add_argument('--meta_feature_type', type=str, default='tabpfn', help='tsfel, tsfel_gaussianRandomProjection, tsfused, tabpfn')
    parser.add_argument('--window_len', type=int, default=50, help='window_len of tabpfn')
    parser.add_argument('--n_samples', type=int, default=None, help='number of samples for tabpfn')
    parser.add_argument('--samples_per_feature', type=int, default=100, help='samples per feature for dynamic n_samples calculation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()

    # Root path for datasets
    root_path = args.root_path

    # Filter dataset directories
    # dataset_dir = [x for x in os.listdir(root_path) if 'plots_multivariate' not in x]
    root_dir = Path(root_path)

    meta_feature_type = args.meta_feature_type # tsfel, tsfel_gaussianRandomProjection, tsfused, tabpfn
    print(f"Meta Feature Type:{meta_feature_type}. n_samples: {args.n_samples}, seed: {args.seed}, window_len: {args.window_len}, samples_per_feature: {args.samples_per_feature}")
    
    if args.dataset == 'm4':
        get_meta_features_m4(root_path=args.root_path, meta_feature_type=meta_feature_type, 
                             seed=args.seed) # ignored other args as they are inferred or unused in new m4 logic
    else:
        # Collect all CSV file paths
        file_paths = [
            str(p) for p in root_dir.rglob('*')
            if p.is_file() and str(p).endswith('.csv') and 'plots' not in str(p) and 'm4' not in str(p) and '00' not in str(p)
        ]
        print(f"Found {len(file_paths)} dataset files for meta-feature extraction.")
        print(file_paths)
        
        # Iterate over files and extract meta-features
        meta_features = get_meta_faetures(file_paths=file_paths, meta_feature_type=meta_feature_type, save_path=None, n_samples=args.n_samples, seed=args.seed, window_len=args.window_len, samples_per_feature=args.samples_per_feature)