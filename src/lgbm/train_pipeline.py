import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle
import json
from typing import Dict, Tuple, List

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
src_path = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(src_path))

from utils.preprocessing import preprocess_data
from utils.data_extractor import extract_data


class LGBMPipeline:
    def __init__(self, config_path: str):
        """
        LightGBMパイプラインの初期化
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.scaler = None
        self.models = {}
        self.results = {}
    
    def load_config(self) -> Dict:
        """設定ファイルを読み込む"""
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def get_data_folder_name(self) -> str:
        """learning_data_numに基づくデータフォルダ名を取得"""
        learning_num = self.config['data_extraction']['learning_data_num']
        validation_num = self.config['data_extraction']['validation_data_num']
        return f"learning_{learning_num}_validation_{validation_num}"
    
    def get_output_path(self) -> str:
        """learning_data_numに応じた出力パスを取得"""
        base_output_path = self.config['data_paths']['output_path']
        data_folder_name = self.get_data_folder_name()
        return os.path.join(base_output_path, data_folder_name)
    
    def step1_preprocessing(self) -> None:
        """ステップ1: 前処理（preprocessing.pyを使用）"""
        print("=== Step 1: Data Preprocessing ===")
        
        # learning_data_numに応じた出力パス
        output_path = self.get_output_path()
        os.makedirs(output_path, exist_ok=True)
        
        target_file = os.path.join(
            output_path,
            self.config['preprocessing']['preprocessed_filename']
        )
        
        # ファイルが存在する場合はスキップ
        if os.path.exists(target_file):
            print(f"Target file already exists: {target_file}")
            print("Skipping preprocessing step.")
            return
        
        # preprocessing.pyの設定を準備
        preprocessing_config = {
            'raw_filenames': self.config['preprocessing']['raw_filenames'],
            'preprocessed_filename': self.config['preprocessing']['preprocessed_filename'],
            'input_path': self.config['data_paths']['input_path'],
            'output_path': output_path,  # learning_data_numに応じたパス
            'columns': self.config['preprocessing']['columns']
        }
        
        print(f"Running preprocessing with output to: {output_path}")
        preprocess_data(preprocessing_config)
        print("Preprocessing completed.\n")
    
    def step2_data_extraction(self) -> None:
        """ステップ2: データ抽出（data_extractor.pyを使用）"""
        print("=== Step 2: Data Extraction ===")
        
        # learning_data_numに応じた出力パス
        output_path = self.get_output_path()
        
        extraction_configs = []
        pattern_names = []
        
        # 各抽出パターンの設定を準備
        for pattern_config in self.config['data_extraction']['extraction_patterns']:
            pattern_name = pattern_config['name']
            pattern_names.append(pattern_name)
            
            # 出力ファイル名を確認
            learning_file = os.path.join(
                output_path,
                f"{pattern_name}_learning.csv"
            )
            validate_file = os.path.join(
                output_path,
                f"{pattern_name}_validate.csv"
            )
            
            # ファイルが存在する場合はスキップ
            if os.path.exists(learning_file) and os.path.exists(validate_file):
                print(f"Files already exist for {pattern_name}: skipping extraction")
                continue
            
            # data_extractor.pyの設定を準備
            extraction_config = {
                'input_filename': self.config['data_extraction']['input_filename'],
                'output_filename': 'dummy.csv',  # 使用されない
                'input_path': output_path,  # learning_data_numに応じたパス
                'output_path': output_path,  # learning_data_numに応じたパス
                'random_seed': self.config['data_extraction']['random_seed'],
                'extraction_pattern': pattern_config['pattern'],
                'learning_data_num': self.config['data_extraction']['learning_data_num'],
                'validation_data_num': self.config['data_extraction']['validation_data_num']
            }
            
            print(f"Extracting data for {pattern_name}...")
            extract_data(extraction_config)
            
            # 出力ファイル名を変更
            default_learning = os.path.join(output_path, 'learning_output.csv')
            default_validate = os.path.join(output_path, 'test_output.csv')
            
            if os.path.exists(default_learning):
                os.rename(default_learning, learning_file)
                print(f"Renamed learning file to: {learning_file}")
            
            if os.path.exists(default_validate):
                os.rename(default_validate, validate_file)
                print(f"Renamed validation file to: {validate_file}")
        
        print("Data extraction completed.\n")
    
    def load_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """データセットを読み込む"""
        print("=== Loading Datasets ===")
        datasets = {}
        
        # learning_data_numに応じた出力パス
        output_path = self.get_output_path()
        
        for pattern_config in self.config['data_extraction']['extraction_patterns']:
            pattern_name = pattern_config['name']
            
            learning_file = os.path.join(
                output_path,
                f"{pattern_name}_learning.csv"
            )
            validate_file = os.path.join(
                output_path,
                f"{pattern_name}_validate.csv"
            )
            
            if os.path.exists(learning_file) and os.path.exists(validate_file):
                datasets[pattern_name] = {
                    'train': pd.read_csv(learning_file),
                    'validation': pd.read_csv(validate_file)
                }
                print(f"Loaded {pattern_name}: train={len(datasets[pattern_name]['train'])}, validation={len(datasets[pattern_name]['validation'])}")
            else:
                print(f"Warning: Missing files for {pattern_name}")
        
        print("Dataset loading completed.\n")
        return datasets
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特徴量と目的変数を準備"""
        features = self.config['machine_learning']['features']
        target = self.config['machine_learning']['target']
        
        X = df[features].copy()
        y = df[target].copy()
        
        return X, y
    
    def apply_standardization(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """標準化を適用"""
        if not self.config['machine_learning']['standardization']['apply']:
            return X_train, X_val
        
        method = self.config['machine_learning']['standardization']['method']
        
        if method == 'StandardScaler':
            self.scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        return X_train_scaled, X_val_scaled
    
    def train_lgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series, pattern_name: str) -> lgb.LGBMRegressor:
        """LightGBMモデルを学習"""
        print(f"Training LightGBM model for {pattern_name}...")
        
        # LightGBMパラメータを取得
        lgbm_params = self.config['machine_learning']['lgbm_params'].copy()
        training_params = self.config['machine_learning']['training']
        
        # モデル作成
        model = lgb.LGBMRegressor(**lgbm_params)
        
        # 学習
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(training_params['early_stopping_rounds']),
                lgb.log_evaluation(training_params['verbose_eval'])
            ]
        )
        
        print(f"Model training completed for {pattern_name}")
        return model
    
    def evaluate_model(self, model: lgb.LGBMRegressor, X_val: pd.DataFrame, 
                      y_val: pd.Series, pattern_name: str) -> Dict[str, float]:
        """モデルを評価"""
        print(f"Evaluating model for {pattern_name}...")
        
        # 予測
        y_pred = model.predict(X_val)
        
        # RMSE計算
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        results = {
            'RMSE': rmse,
            'n_samples': len(y_val)
        }
        
        print(f"Results for {pattern_name}: RMSE = {rmse:.4f}")
        
        # 予測結果を保存（設定で有効な場合）
        if self.config['evaluation']['save_predictions']:
            predictions_df = pd.DataFrame({
                'actual': y_val.values,
                'predicted': y_pred,
                'error': y_val.values - y_pred
            })
            
            # learning_data_numに応じた出力パス
            output_path = self.get_output_path()
            pred_file = os.path.join(
                output_path,
                f"{pattern_name}_predictions.csv"
            )
            predictions_df.to_csv(pred_file, index=False)
            print(f"Predictions saved to: {pred_file}")
        
        return results
    
    def save_model(self, model: lgb.LGBMRegressor, pattern_name: str) -> None:
        """モデルを保存"""
        # learning_data_numに応じたモデル保存パス
        learning_num = self.config['data_extraction']['learning_data_num']
        validation_num = self.config['data_extraction']['validation_data_num']
        model_dir = os.path.join(
            self.config['data_paths']['model_path'],
            f"learning_{learning_num}_validation_{validation_num}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(
            model_dir,
            f"{pattern_name}_lgbm_model.pkl"
        )
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model saved to: {model_file}")
    
    def save_scaler(self) -> None:
        """スケーラーを保存"""
        if self.scaler is not None:
            # learning_data_numに応じたスケーラー保存パス
            learning_num = self.config['data_extraction']['learning_data_num']
            validation_num = self.config['data_extraction']['validation_data_num']
            model_dir = os.path.join(
                self.config['data_paths']['model_path'],
                f"learning_{learning_num}_validation_{validation_num}"
            )
            os.makedirs(model_dir, exist_ok=True)
            
            scaler_file = os.path.join(
                model_dir,
                "scaler.pkl"
            )
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"Scaler saved to: {scaler_file}")
    
    def run_pipeline(self) -> None:
        """パイプライン全体を実行"""
        print("Starting LightGBM Machine Learning Pipeline...\n")
        
        # ステップ1: 前処理
        self.step1_preprocessing()
        
        # ステップ2: データ抽出
        self.step2_data_extraction()
        
        # データセット読み込み
        datasets = self.load_datasets()
        
        if not datasets:
            print("No datasets loaded. Exiting pipeline.")
            return
        
        print("=== Step 3: Model Training and Evaluation ===")
        
        # 各抽出パターンに対してモデルを学習・評価
        for pattern_name, data in datasets.items():
            print(f"\n--- Processing {pattern_name.upper()} dataset ---")
            
            # 特徴量と目的変数を準備
            X_train, y_train = self.prepare_features_and_target(data['train'])
            X_val, y_val = self.prepare_features_and_target(data['validation'])
            
            print(f"Features: {list(X_train.columns)}")
            print(f"Target: {self.config['machine_learning']['target']}")
            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            
            # 標準化
            X_train_processed, X_val_processed = self.apply_standardization(X_train, X_val)
            
            if self.scaler is not None:
                print(f"Applied standardization: {self.config['machine_learning']['standardization']['method']}")
            
            # モデル学習
            model = self.train_lgbm_model(X_train_processed, y_train, X_val_processed, y_val, pattern_name)
            
            # モデル評価
            results = self.evaluate_model(model, X_val_processed, y_val, pattern_name)
            
            # 結果を保存
            self.models[pattern_name] = model
            self.results[pattern_name] = results
            
            # モデル保存
            self.save_model(model, pattern_name)
        
        # スケーラー保存
        self.save_scaler()
        
        # 最終結果表示
        self.display_final_results()
        
        # 結果をJSONで保存
        self.save_results()
        
        print("\nPipeline completed successfully!")
    
    def display_final_results(self) -> None:
        """最終結果を表示"""
        print("\n=== Final Results Summary ===")
        print("Pattern\t\tRMSE\t\tSamples")
        print("-" * 40)
        
        for pattern_name, results in self.results.items():
            print(f"{pattern_name:<15}\t{results['RMSE']:.4f}\t\t{results['n_samples']}")
    
    def save_results(self) -> None:
        """結果をJSONファイルに保存"""
        # learning_data_numに応じた出力パス
        output_path = self.get_output_path()
        results_file = os.path.join(
            output_path,
            "training_results.json"
        )
        
        # 結果を保存用に準備
        save_results = {
            'config': self.config_path,
            'data_folder': self.get_data_folder_name(),
            'results': self.results,
            'model_info': {
                'features': self.config['machine_learning']['features'],
                'target': self.config['machine_learning']['target'],
                'standardization': self.config['machine_learning']['standardization'],
                'lgbm_params': self.config['machine_learning']['lgbm_params'],
                'learning_data_num': self.config['data_extraction']['learning_data_num'],
                'validation_data_num': self.config['data_extraction']['validation_data_num']
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {results_file}")


def main():
    """メイン関数"""
    try:
        # 設定ファイルのパス
        config_path = Path(__file__).parent / "config" / "config.yaml"
        
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            return
        
        print(f"Loading config from: {config_path}")
        
        # パイプライン実行
        pipeline = LGBMPipeline(str(config_path))
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()