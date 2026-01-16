import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, List
import logging
import os
import sys
from pathlib import Path

# 既存のutils関数をインポート
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.preprocessing import preprocess_data
from src.utils.data_extractor import extract_data


class RegressionDataset(Dataset):
    """
    回帰問題用のPyTorchデータセット
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 目的変数データ
        """
        # データクリーニング
        X, y = self._clean_data(X, y)
        
        # データ型を確保
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
    
    def _clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        データの不正値を除去
        
        Args:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 目的変数データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: クリーニング済みデータ
        """
        # 文字列や不正値のマスクを作成
        def is_numeric_convertible(arr):
            """配列が数値に変換可能かチェック"""
            if arr.dtype == 'object':
                try:
                    # 各要素を個別にチェック
                    mask = np.zeros(len(arr), dtype=bool)
                    for i, val in enumerate(arr):
                        try:
                            float(str(val))
                            # '#NAME?', 'N/A', '', 'nan'などの不正値もチェック
                            val_str = str(val).strip().upper()
                            if val_str in ['#NAME?', '#N/A', '#REF!', '#DIV/0!', '', 'NAN', 'INF', '-INF']:
                                mask[i] = False
                            else:
                                mask[i] = True
                        except (ValueError, TypeError):
                            mask[i] = False
                    return mask
                except:
                    return np.zeros(len(arr), dtype=bool)
            else:
                # 数値型の場合は有限値をチェック
                return np.isfinite(arr)
        
        # X (特徴量) の各列をチェック
        X_mask = np.ones(X.shape[0], dtype=bool)
        if X.ndim > 1:
            for col in range(X.shape[1]):
                X_mask &= is_numeric_convertible(X[:, col])
        else:
            X_mask = is_numeric_convertible(X)
        
        # y (目的変数) をチェック
        y_mask = is_numeric_convertible(y)
        
        # 両方とも有効な行のみを保持
        valid_mask = X_mask & y_mask
        
        if not np.any(valid_mask):
            raise ValueError("No valid numeric data found after cleaning")
        
        # 無効な行が見つかった場合の警告
        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            print(f"Warning: {invalid_count} rows with invalid data removed")
        
        return X[valid_mask], y[valid_mask]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataManager:
    """
    データ管理クラス - Source/Targetデータの読み込み・前処理・分割を管理
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # スケーラーの初期化
        self.scaler = None
        self.init_scaler()
        
        # データ格納用
        self.source_data = None
        self.target_data = None
        
        # 前処理・抽出済みデータの保存先
        self.output_path = None
        
    def init_scaler(self):
        """スケーラーの初期化"""
        if self.config['model_data']['standardization']['apply']:
            method = self.config['model_data']['standardization']['method']
            if method == 'StandardScaler':
                self.scaler = StandardScaler()
            elif method == 'MinMaxScaler':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler method: {method}")
        else:
            self.scaler = None
    
    def setup_output_path(self):
        """データ抽出用の出力パスを設定"""
        learning_num = self.config['data_extraction']['learning_data_num']
        validation_num = self.config['data_extraction']['validation_data_num']
        data_folder_name = f"learning_{learning_num}_validation_{validation_num}"
        self.output_path = os.path.join(self.config['data_paths']['output_path'], data_folder_name)
        os.makedirs(self.output_path, exist_ok=True)
        return self.output_path
    
    def run_preprocessing(self, data_type: str) -> str:
        """
        既存のpreprocessing.pyを使用して前処理を実行
        
        Args:
            data_type (str): 'source' or 'target'
            
        Returns:
            str: 前処理済みファイルのパス
        """
        self.logger.info(f"Running preprocessing for {data_type} data...")
        
        # 設定準備
        if data_type == 'source':
            raw_filenames = self.config['preprocessing']['source_raw_filenames']
            preprocessed_filename = self.config['preprocessing']['source_preprocessed_filename']
        else:  # target
            raw_filenames = self.config['preprocessing']['target_raw_filenames']
            preprocessed_filename = self.config['preprocessing']['target_preprocessed_filename']
        
        # 前処理済みファイルのパス
        preprocessed_path = os.path.join(self.config['data_paths']['preprocessed_path'])
        os.makedirs(preprocessed_path, exist_ok=True)
        
        preprocessed_file_path = os.path.join(preprocessed_path, preprocessed_filename)
        
        # ファイルが存在する場合はスキップ
        if os.path.exists(preprocessed_file_path):
            self.logger.info(f"Preprocessed file already exists: {preprocessed_file_path}")
            return preprocessed_file_path
        
        # preprocessing.pyの設定
        preprocessing_config = {
            'raw_filenames': raw_filenames,
            'preprocessed_filename': preprocessed_filename,
            'input_path': self.config['preprocessing']['input_path'],
            'output_path': preprocessed_path,
            'columns': self.config['preprocessing']['columns']
        }
        
        # 前処理実行
        preprocess_data(preprocessing_config)
        self.logger.info(f"Preprocessing completed: {preprocessed_file_path}")
        
        return preprocessed_file_path
    
    def run_data_extraction(self, preprocessed_file_path: str, pattern_name: str = None) -> Tuple[str, str]:
        """
        既存のdata_extractor.pyを使用してデータ抽出を実行
        
        Args:
            preprocessed_file_path (str): 前処理済みファイルのパス
            pattern_name (str): 抽出パターン名（Noneの場合はデフォルト使用）
            
        Returns:
            Tuple[str, str]: (学習データファイルパス, 検証データファイルパス)
        """
        if pattern_name is None:
            pattern_name = self.config['data_extraction']['default_pattern']
        
        self.logger.info(f"Running data extraction with pattern: {pattern_name}")
        
        # 出力パスの設定
        output_path = self.setup_output_path()
        
        # 抽出済みファイルのパス
        learning_file = os.path.join(output_path, f"{pattern_name}_learning.csv")
        validation_file = os.path.join(output_path, f"{pattern_name}_validate.csv")
        
        # ファイルが存在する場合はスキップ
        if os.path.exists(learning_file) and os.path.exists(validation_file):
            self.logger.info(f"Extracted files already exist: {learning_file}, {validation_file}")
            return learning_file, validation_file
        
        # パターン番号の取得
        pattern_mapping = {
            'random': 1,
            'interpolation': 2,
            'extrapolation': 3
        }
        pattern_number = pattern_mapping.get(pattern_name, 2)  # デフォルトは interpolation
        
        # data_extractor.pyの設定
        extraction_config = {
            'input_filename': os.path.basename(preprocessed_file_path),
            'output_filename': 'dummy.csv',  # 使用されない
            'input_path': os.path.dirname(preprocessed_file_path),
            'output_path': output_path,
            'random_seed': self.config['data_extraction']['random_seed'],
            'extraction_pattern': pattern_number,
            'learning_data_num': self.config['data_extraction']['learning_data_num'],
            'validation_data_num': self.config['data_extraction']['validation_data_num']
        }
        
        # データ抽出実行
        extract_data(extraction_config)
        
        # ファイル名の変更
        default_learning = os.path.join(output_path, 'learning_output.csv')
        default_validate = os.path.join(output_path, 'test_output.csv')
        
        if os.path.exists(default_learning):
            os.rename(default_learning, learning_file)
        if os.path.exists(default_validate):
            os.rename(default_validate, validation_file)
        
        self.logger.info(f"Data extraction completed: {learning_file}, {validation_file}")
        
        return learning_file, validation_file
    
    def load_extracted_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        抽出済みデータファイルを読み込み
        
        Args:
            file_path (str): CSVファイルのパス
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特徴量, 目的変数)
        """
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded extracted data from {file_path}: {len(df)} samples")
        
        # 特徴量と目的変数の抽出
        features = self.config['model_data']['features']
        target = self.config['model_data']['target']
        
        X = df[features].values
        y = df[target].values
        
        self.logger.info(f"Features: {features}")
        self.logger.info(f"Target: {target}")
        self.logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        
        # 欠損値のチェック
        if np.isnan(X).any() or np.isnan(y).any():
            self.logger.warning("Missing values detected in data")
            # 欠損値を含む行を除去
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            self.logger.info(f"After removing missing values: X={X.shape}, y={y.shape}")
        
        # データの統計情報出力
        self.logger.info(f"Data statistics:")
        self.logger.info(f"  Features (X): min={X.min(axis=0)}, max={X.max(axis=0)}, mean={X.mean(axis=0)}")
        self.logger.info(f"  Target (y): min={y.min()}, max={y.max()}, mean={y.mean()}")
        
        return X, y
    
    def prepare_source_data(self, pattern_name: str = None) -> Dict[str, DataLoader]:
        """
        Sourceデータの準備（全数の8:2で分割）
        
        Args:
            pattern_name (str): 抽出パターン名（source設定を使用）
            
        Returns:
            Dict[str, DataLoader]: SourceデータのDataLoader
        """
        self.logger.info("Preparing source data (FeCu)...")
        
        # 1. 前処理実行
        preprocessed_file = self.run_preprocessing('source')
        
        # 2. Sourceデータの8:2分割
        X_train, y_train, X_val, y_val = self.split_source_data(preprocessed_file)
        
        # 3. スケーリング
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            self.logger.info("Applied scaling to source data")
        
        # 4. DataLoader作成
        batch_size = self.config['pretrain']['batch_size']
        dataloaders = self.create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
        
        # データを保存（評価用）
        self.source_data = (X_train, y_train, X_val, y_val)
        
        return dataloaders
    
    def split_source_data(self, preprocessed_file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sourceデータを8:2で分割
        
        Args:
            preprocessed_file_path (str): 前処理済みファイルのパス
            
        Returns:
            Tuple: (X_train, y_train, X_val, y_val)
        """
        # データ読み込み
        df = pd.read_csv(preprocessed_file_path)
        
        # 特徴量と目的変数の分離
        features = self.config['model_data']['features']
        target = self.config['model_data']['target']
        
        X = df[features].values
        y = df[target].values
        
        # 8:2分割
        train_ratio = self.config['data_extraction']['source']['train_ratio']
        random_seed = self.config['data_extraction']['random_seed']
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(df))
        
        split_idx = int(len(df) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        self.logger.info(f"Source data split: Train={len(X_train)}, Validation={len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def prepare_target_data(self, pattern_name: str = None) -> Dict[str, DataLoader]:
        """
        Targetデータの準備（指定数の学習データ + 2割の検証データ）
        
        Args:
            pattern_name (str): 検証データの抽出パターン名
            
        Returns:
            Dict[str, DataLoader]: TargetデータのDataLoader
        """
        self.logger.info("Preparing target data (FeFe)...")
        
        # 1. 前処理実行
        preprocessed_file = self.run_preprocessing('target')
        
        # 2. Targetデータの抽出（学習データ指定数 + 検証データ全数の2割）
        X_train, y_train, X_val, y_val = self.extract_target_data(preprocessed_file, pattern_name)
        
        # 3. スケーリング（Sourceデータで学習済みのスケーラーを使用）
        if self.scaler is not None:
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)
            self.logger.info("Applied scaling to target data with source scaler")
        
        # 4. DataLoader作成
        batch_size = self.config['finetune']['batch_size']
        dataloaders = self.create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
        
        # データを保存（評価用）
        self.target_data = (X_train, y_train, X_val, y_val)
        
        return dataloaders
    
    def extract_target_data(self, preprocessed_file_path: str, pattern_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Targetデータから学習用（指定数）と検証用（2割、パターン指定）を抽出
        
        Args:
            preprocessed_file_path (str): 前処理済みファイルのパス
            pattern_name (str): 検証データの抽出パターン名
            
        Returns:
            Tuple: (X_train, y_train, X_val, y_val)
        """
        # データ読み込み
        df = pd.read_csv(preprocessed_file_path)
        
        # 設定取得
        target_config = self.config['data_extraction']['target']
        learning_data_num = target_config['learning_data_num']
        validation_ratio = target_config['validation_ratio']
        validation_pattern = pattern_name or target_config['validation_pattern']
        random_seed = self.config['data_extraction']['random_seed']
        
        # 特徴量と目的変数の分離
        features = self.config['model_data']['features']
        target = self.config['model_data']['target']
        
        X = df[features].values
        y = df[target].values
        
        np.random.seed(random_seed)
        
        # 1. 学習用データの抽出（指定数）
        if learning_data_num >= len(df):
            train_indices = np.arange(len(df))
        else:
            train_indices = np.random.choice(len(df), learning_data_num, replace=False)
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        # 2. 検証用データの抽出（全体の2割、パターン指定）
        # 残りのデータから検証データを抽出
        remaining_indices = np.setdiff1d(np.arange(len(df)), train_indices)
        validation_data_num = int(len(df) * validation_ratio)
        
        if validation_pattern == "random":
            val_indices = np.random.choice(remaining_indices, 
                                         min(validation_data_num, len(remaining_indices)), 
                                         replace=False)
        elif validation_pattern == "interpolation":
            # 学習データの範囲内でランダムサンプリング
            val_indices = self._extract_interpolation_validation(X, y, train_indices, 
                                                               remaining_indices, validation_data_num)
        elif validation_pattern == "extrapolation":
            # 学習データの範囲外でランダムサンプリング
            val_indices = self._extract_extrapolation_validation(X, y, train_indices, 
                                                                remaining_indices, validation_data_num)
        else:
            val_indices = np.random.choice(remaining_indices, 
                                         min(validation_data_num, len(remaining_indices)), 
                                         replace=False)
        
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        self.logger.info(f"Target data extracted: Train={len(X_train)}, Validation={len(X_val)} ({validation_pattern})")
        
        return X_train, y_train, X_val, y_val
    
    def _extract_interpolation_validation(self, X: np.ndarray, y: np.ndarray, 
                                        train_indices: np.ndarray, remaining_indices: np.ndarray, 
                                        validation_data_num: int) -> np.ndarray:
        """
        内挿の検証データを抽出（学習データの特徴量範囲内）
        """
        # 学習データの特徴量範囲
        X_train = X[train_indices]
        min_vals = np.min(X_train, axis=0)
        max_vals = np.max(X_train, axis=0)
        
        # 残りのデータから範囲内のデータを抽出
        X_remaining = X[remaining_indices]
        in_range_mask = np.all((X_remaining >= min_vals) & (X_remaining <= max_vals), axis=1)
        interpolation_indices = remaining_indices[in_range_mask]
        
        # 必要数をランダム選択
        selected_num = min(validation_data_num, len(interpolation_indices))
        if selected_num > 0:
            return np.random.choice(interpolation_indices, selected_num, replace=False)
        else:
            # 範囲内データがない場合はランダムに選択
            return np.random.choice(remaining_indices, 
                                  min(validation_data_num, len(remaining_indices)), 
                                  replace=False)
    
    def _extract_extrapolation_validation(self, X: np.ndarray, y: np.ndarray, 
                                        train_indices: np.ndarray, remaining_indices: np.ndarray, 
                                        validation_data_num: int) -> np.ndarray:
        """
        外挿の検証データを抽出（学習データの特徴量範囲外）
        """
        # 学習データの特徴量範囲
        X_train = X[train_indices]
        min_vals = np.min(X_train, axis=0)
        max_vals = np.max(X_train, axis=0)
        
        # 残りのデータから範囲外のデータを抽出
        X_remaining = X[remaining_indices]
        out_range_mask = np.any((X_remaining < min_vals) | (X_remaining > max_vals), axis=1)
        extrapolation_indices = remaining_indices[out_range_mask]
        
        # 必要数をランダム選択
        selected_num = min(validation_data_num, len(extrapolation_indices))
        if selected_num > 0:
            return np.random.choice(extrapolation_indices, selected_num, replace=False)
        else:
            # 範囲外データがない場合はランダムに選択
            return np.random.choice(remaining_indices, 
                                  min(validation_data_num, len(remaining_indices)), 
                                  replace=False)
    
    
    def apply_scaling(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        スケーリングを適用
        
        Args:
            X_train (np.ndarray): 学習データ
            X_val (np.ndarray, optional): 検証データ
            
        Returns:
            Tuple[np.ndarray, ...]: スケーリング済みデータ
        """
        if self.scaler is None:
            return X_train, X_val
        
        # 学習データでスケーラーを学習
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    def create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None,
                          batch_size: int = 32, 
                          shuffle_train: bool = True) -> Dict[str, DataLoader]:
        """
        PyTorchのDataLoaderを作成
        
        Args:
            X_train (np.ndarray): 学習データ特徴量
            y_train (np.ndarray): 学習データ目的変数
            X_val (np.ndarray, optional): 検証データ特徴量
            y_val (np.ndarray, optional): 検証データ目的変数
            batch_size (int): バッチサイズ
            shuffle_train (bool): 学習データをシャッフルするか
            
        Returns:
            Dict[str, DataLoader]: DataLoaderの辞書
        """
        dataloaders = {}
        
        # 学習データローダー
        train_dataset = RegressionDataset(X_train, y_train)
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train
        )
        
        # 検証データローダー
        if X_val is not None and y_val is not None:
            val_dataset = RegressionDataset(X_val, y_val)
            dataloaders['validation'] = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        return dataloaders
    
    def get_test_data(self, data_type: str = 'both') -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        テストデータを取得（検証データをテストデータとして使用）
        
        Args:
            data_type (str): 'source', 'target', or 'both'
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: テストデータ
        """
        test_data = {}
        
        if data_type in ['source', 'both']:
            if self.source_data is not None:
                # source_dataの形式: (X_train, y_train, X_val, y_val)
                X_test, y_test = self.source_data[2], self.source_data[3]
                test_data['source'] = (X_test, y_test)
        
        if data_type in ['target', 'both']:
            if self.target_data is not None:
                # target_dataの形式: (X_train, y_train, X_val, y_val)
                X_test, y_test = self.target_data[2], self.target_data[3]
                test_data['target'] = (X_test, y_test)
        
        return test_data