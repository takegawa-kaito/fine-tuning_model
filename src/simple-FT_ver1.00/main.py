import os
import sys
import yaml
import torch
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataManager
from utils.models import ModelFactory, init_weights
from utils.trainer import Trainer
from utils.evaluator import Evaluator


def setup_logging(config: dict) -> logging.Logger:
    """
    ログ設定のセットアップ
    
    Args:
        config (dict): 設定辞書
        
    Returns:
        logging.Logger: ロガー
    """
    log_config = config.get('logging', {})
    
    # ログレベル
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # フォーマット
    log_format = log_config.get('format', 
                               '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # ルートロガーの設定
    logging.basicConfig(level=log_level, format=log_format)
    
    logger = logging.getLogger('FineTuning')
    
    # ファイル出力の設定
    if log_config.get('save_to_file', False):
        log_dir = os.path.join(config['data_paths']['output_path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, log_config.get('log_file', 'training.log'))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    return logger


def setup_device(config: dict) -> torch.device:
    """
    デバイス設定のセットアップ
    
    Args:
        config (dict): 設定辞書
        
    Returns:
        torch.device: 使用デバイス
    """
    device_config = config['experiment'].get('device', 'auto')
    
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    return device


def set_random_seeds(seed: int):
    """
    再現性のためのランダムシード設定
    
    Args:
        seed (int): ランダムシード
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_directory(config: dict) -> str:
    """
    実験ディレクトリの作成
    
    Args:
        config (dict): 設定辞書
        
    Returns:
        str: 実験ディレクトリパス
    """
    # タイムスタンプ付きのディレクトリ名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    target_size = config['data_extraction']['target']['learning_data_num']
    exp_dir_name = f"{exp_name}_target{target_size}_{timestamp}"
    
    # 実験ディレクトリのパス
    exp_dir = os.path.join(config['data_paths']['output_path'], 'experiments', exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # サブディレクトリ作成
    sub_dirs = ['models', 'plots', 'results']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(exp_dir, sub_dir), exist_ok=True)
    
    return exp_dir


class FineTuningPipeline:
    """
    ファインチューニングパイプラインのメインクラス
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): 設定ファイルのパス
        """
        # config_pathを保存
        self.config_path = config_path
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ログ設定
        self.logger = setup_logging(self.config)
        self.logger.info("FineTuning Pipeline initialized")
        self.logger.info(f"Config loaded from: {config_path}")
        
        # デバイス設定
        self.device = setup_device(self.config)
        self.logger.info(f"Using device: {self.device}")
        
        # ランダムシード設定
        random_seed = self.config['experiment']['random_seed']
        set_random_seeds(random_seed)
        self.logger.info(f"Random seed set to: {random_seed}")
        
        # 実験ディレクトリ作成
        self.exp_dir = create_experiment_directory(self.config)
        self.logger.info(f"Experiment directory created: {self.exp_dir}")
        
        # データマネージャー初期化
        self.data_manager = DataManager(self.config)
        
        # モデル作成
        self.model = ModelFactory.create_model(self.config)
        self.model.apply(init_weights)  # 重みを初期化
        
        # パラメータ数をログ出力
        param_info = self.model.get_num_parameters()
        self.logger.info(f"Model parameters: {param_info}")
        
        # トレーナー・評価器初期化
        self.trainer = Trainer(self.model, self.config, self.device)
        self.evaluator = Evaluator(self.config)
        
        # 結果保存用
        self.results = {}
    
    def run_pretraining(self) -> dict:
        """
        事前学習（Sourceデータ）を実行
        
        Returns:
            dict: 事前学習結果
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING PRETRAINING PHASE (Source Domain)")
        self.logger.info("=" * 50)
        
        # Sourceデータの準備
        pattern_name = self.config['data_extraction'].get('default_pattern', 'interpolation')
        source_dataloaders = self.data_manager.prepare_source_data(pattern_name)
        
        # 事前学習実行
        pretrain_results = self.trainer.train(
            source_dataloaders['train'],
            source_dataloaders['validation'],
            config_key='pretrain'
        )
        
        # モデル保存
        pretrain_model_path = os.path.join(self.exp_dir, 'models', 'pretrained_model.pth')
        self.trainer.save_model(pretrain_model_path, {
            'phase': 'pretrain',
            'experiment_config': self.config
        })
        
        # 学習曲線保存
        curves_path = os.path.join(self.exp_dir, 'plots', 'pretrain_curves.png')
        self.trainer.plot_training_curves(curves_path)
        
        self.logger.info("Pretraining phase completed")
        
        return pretrain_results
    
    def run_finetuning(self) -> dict:
        """
        ファインチューニング（Targetデータ）を実行
        
        Returns:
            dict: ファインチューニング結果
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING FINETUNING PHASE (Target Domain)")
        self.logger.info("=" * 50)
        
        # Targetデータの準備
        pattern_name = self.config['data_extraction'].get('default_pattern', 'interpolation')
        target_dataloaders = self.data_manager.prepare_target_data(pattern_name)
        
        # 層の凍結設定
        freeze_config = self.config['finetune']['freeze_layers']
        if freeze_config['enable']:
            self.model.freeze_layers(freeze_config['freeze_until_layer'])
            self.logger.info(f"Froze {freeze_config['freeze_until_layer']} layers for finetuning")
        
        # パラメータ数の再確認
        param_info = self.model.get_num_parameters()
        self.logger.info(f"Finetuning parameters: {param_info}")
        
        # ファインチューニング実行
        finetune_results = self.trainer.train(
            target_dataloaders['train'],
            target_dataloaders['validation'],
            config_key='finetune'
        )
        
        # モデル保存
        finetune_model_path = os.path.join(self.exp_dir, 'models', 'finetuned_model.pth')
        self.trainer.save_model(finetune_model_path, {
            'phase': 'finetune',
            'experiment_config': self.config
        })
        
        # 学習曲線保存（累積）
        curves_path = os.path.join(self.exp_dir, 'plots', 'finetune_curves.png')
        self.trainer.plot_training_curves(curves_path)
        
        self.logger.info("Finetuning phase completed")
        
        return finetune_results
    
    def evaluate_models(self) -> dict:
        """
        モデルの評価を実行（Source/Target別評価）
        
        Returns:
            dict: 評価結果 
        """
        self.logger.info("=" * 50)
        self.logger.info("STARTING EVALUATION PHASE")
        self.logger.info("=" * 50)
        
        evaluation_results = {
            'pretrain': {'source': {}},
            'finetune': {'source': {}, 'target': {}}
        }
        
        # テストデータの取得
        test_data = self.data_manager.get_test_data('both')
        
        # 1. 事前学習モデルでの評価（Sourceデータのみ）
        self.logger.info("Evaluating pretrained model on source domain...")
        pretrained_model_path = os.path.join(self.exp_dir, 'models', 'pretrained_model.pth')
        if os.path.exists(pretrained_model_path):
            # 事前学習モデルを読み込み
            pretrained_state = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(pretrained_state['model_state_dict'])
            
            # Sourceデータで評価
            if 'source' in test_data:
                X_test_source, y_test_source = test_data['source']
                source_metrics = self._evaluate_on_data(X_test_source, y_test_source, "Pretrain-Source")
                evaluation_results['pretrain']['source'] = source_metrics
                
                # プロット生成
                pred_plot_path = os.path.join(self.exp_dir, 'plots', 'pretrain_source_predictions.png')
                residuals_plot_path = os.path.join(self.exp_dir, 'plots', 'pretrain_source_residuals.png')
                self._generate_evaluation_plots(X_test_source, y_test_source, 
                                              "Pretrained Model - Source Domain (FeCu)", 
                                              pred_plot_path, residuals_plot_path)
        
        # 2. ファインチューニング後モデルでの評価（Source + Target）
        self.logger.info("Evaluating finetuned model on both domains...")
        finetuned_model_path = os.path.join(self.exp_dir, 'models', 'finetuned_model.pth')
        if os.path.exists(finetuned_model_path):
            # ファインチューニング後モデルを読み込み
            finetuned_state = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(finetuned_state['model_state_dict'])
            
            # Sourceデータで評価
            if 'source' in test_data:
                X_test_source, y_test_source = test_data['source']
                source_metrics = self._evaluate_on_data(X_test_source, y_test_source, "Finetune-Source")
                evaluation_results['finetune']['source'] = source_metrics
                
                # プロット生成
                pred_plot_path = os.path.join(self.exp_dir, 'plots', 'finetune_source_predictions.png')
                residuals_plot_path = os.path.join(self.exp_dir, 'plots', 'finetune_source_residuals.png')
                self._generate_evaluation_plots(X_test_source, y_test_source, 
                                              "Finetuned Model - Source Domain (FeCu)", 
                                              pred_plot_path, residuals_plot_path)
            
            # Targetデータで評価
            if 'target' in test_data:
                X_test_target, y_test_target = test_data['target']
                target_metrics = self._evaluate_on_data(X_test_target, y_test_target, "Finetune-Target")
                evaluation_results['finetune']['target'] = target_metrics
                
                # プロット生成
                pred_plot_path = os.path.join(self.exp_dir, 'plots', 'finetune_target_predictions.png')
                residuals_plot_path = os.path.join(self.exp_dir, 'plots', 'finetune_target_residuals.png')
                self._generate_evaluation_plots(X_test_target, y_test_target, 
                                              "Finetuned Model - Target Domain (FeFe)", 
                                              pred_plot_path, residuals_plot_path)
        
        # 3. ドメイン間・モデル間比較分析
        if 'source' in test_data and 'target' in test_data:
            domain_plot_path = os.path.join(self.exp_dir, 'plots', 'domain_analysis.png')
            self.evaluator.plot_domain_analysis(
                evaluation_results['finetune']['source'], evaluation_results['finetune']['target'],
                test_data['source'], test_data['target'],
                domain_plot_path
            )
        
        # 特徴量重要度分析
        feature_names = self.config['model_data']['features']
        feature_plot_path = os.path.join(self.exp_dir, 'plots', 'feature_importance.png')
        self.evaluator.plot_feature_importance(self.model, feature_names, 
                                             "Feature Importance Analysis", 
                                             feature_plot_path)
        
        # モデル比較（事前学習vs事後学習）
        model_comparison = {
            'Pretrained_Source': evaluation_results['pretrain']['source'],
            'Finetuned_Source': evaluation_results['finetune']['source'],
            'Finetuned_Target': evaluation_results['finetune']['target']
        }
        
        comparison_plot_path = os.path.join(self.exp_dir, 'plots', 'model_comparison.png')
        self.evaluator.compare_models(model_comparison, 
                                    "Model Performance Comparison (Pretrain vs Finetune)", 
                                    comparison_plot_path)
        
        self.logger.info("Evaluation phase completed")
        
        # ログに詳細結果を出力
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 50)
        
        if evaluation_results['pretrain']['source']:
            self.logger.info(f"Pretrained model (Source): RMSE = {evaluation_results['pretrain']['source'].get('RMSE', 'N/A'):.4f}")
        
        if evaluation_results['finetune']['source']:
            self.logger.info(f"Finetuned model (Source): RMSE = {evaluation_results['finetune']['source'].get('RMSE', 'N/A'):.4f}")
        
        if evaluation_results['finetune']['target']:
            self.logger.info(f"Finetuned model (Target): RMSE = {evaluation_results['finetune']['target'].get('RMSE', 'N/A'):.4f}")
        
        return evaluation_results
    
    def _evaluate_on_data(self, X_test: np.ndarray, y_test: np.ndarray, eval_name: str) -> dict:
        """
        指定データでのモデル評価
        
        Args:
            X_test (np.ndarray): テスト特徴量
            y_test (np.ndarray): テスト目的変数
            eval_name (str): 評価名
            
        Returns:
            dict: 評価メトリクス
        """
        # DataLoader作成
        test_dataset = self.data_manager.create_dataloaders(
            X_test, y_test, batch_size=64, shuffle_train=False
        )['train']  # テストなのでtrainキーを使用
        
        # 予測実行
        y_pred, y_true = self.trainer.predict(test_dataset)
        
        # メトリクス計算
        metrics = self.evaluator.calculate_detailed_metrics(y_true, y_pred)
        
        self.logger.info(f"{eval_name} evaluation - RMSE: {metrics.get('RMSE', 'N/A'):.4f}, R2: {metrics.get('R2', 'N/A'):.4f}")
        
        return metrics
    
    def _generate_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray, 
                                 title: str, pred_plot_path: str, residuals_plot_path: str):
        """
        評価プロット生成
        
        Args:
            X_test (np.ndarray): テスト特徴量
            y_test (np.ndarray): テスト目的変数
            title (str): プロットタイトル
            pred_plot_path (str): 予測プロット保存パス
            residuals_plot_path (str): 残差プロット保存パス
        """
        # DataLoader作成
        test_dataset = self.data_manager.create_dataloaders(
            X_test, y_test, batch_size=64, shuffle_train=False
        )['train']
        
        # 予測実行
        y_pred, y_true = self.trainer.predict(test_dataset)
        
        # 予測結果プロット
        self.evaluator.plot_predictions(y_true, y_pred, 
                                      f"{title} - Predictions vs Actual", 
                                      pred_plot_path)
        
        # 残差分析プロット
        self.evaluator.plot_residuals_analysis(y_true, y_pred,
                                              f"{title} - Residuals Analysis",
                                              residuals_plot_path)
    
    def save_final_results(self, pretrain_results: dict, finetune_results: dict, 
                          evaluation_results: dict) -> None:
        """
        最終結果を保存
        
        Args:
            pretrain_results (dict): 事前学習結果
            finetune_results (dict): ファインチューニング結果
            evaluation_results (dict): 評価結果
        """
        # 全結果をまとめる
        final_results = {
            'experiment_info': {
                'name': self.config['experiment']['name'],
                'description': self.config['experiment']['description'],
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'experiment_directory': self.exp_dir
            },
            'config': self.config,
            'pretrain_results': pretrain_results,
            'finetune_results': finetune_results,
            'evaluation_results': evaluation_results
        }
        
        # JSON形式で保存
        results_json_path = os.path.join(self.exp_dir, 'results', 'final_results.json')
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV形式のサマリーを保存
        results_csv_path = os.path.join(self.exp_dir, 'results', 'results_summary.csv')
        summary_data = {
            'pretrain': pretrain_results,
            'finetune': finetune_results,
            **evaluation_results
        }
        self.evaluator.save_results_summary(summary_data, results_csv_path)
        
        self.logger.info(f"Final results saved to: {results_json_path}")
        self.logger.info(f"Results summary saved to: {results_csv_path}")
        
        # 設定ファイルもコピー保存
        import shutil
        config_copy_path = os.path.join(self.exp_dir, 'config.yaml')
        shutil.copy2(self.config_path, config_copy_path)
        self.logger.info(f"Config file copied to: {config_copy_path}")
    
    def run_full_pipeline(self) -> dict:
        """
        フル パイプラインを実行
        
        Returns:
            dict: 全結果
        """
        try:
            self.logger.info("Starting Full FineTuning Pipeline")
            self.logger.info(f"Experiment: {self.config['experiment']['name']}")
            self.logger.info(f"Description: {self.config['experiment']['description']}")
            
            # 1. 事前学習
            pretrain_results = self.run_pretraining()
            
            # 2. ファインチューニング
            finetune_results = self.run_finetuning()
            
            # 3. 評価
            evaluation_results = self.evaluate_models()
            
            # 4. 結果保存
            self.save_final_results(pretrain_results, finetune_results, evaluation_results)
            
            # サマリー出力
            self.logger.info("=" * 50)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 50)
            
            # Source/Target別でのRMSE結果サマリー
            if evaluation_results['pretrain']['source']:
                pretrain_source_rmse = evaluation_results['pretrain']['source']['RMSE']
                self.logger.info(f"Pretrained Model (Source): RMSE = {pretrain_source_rmse:.4f}")
            
            if evaluation_results['finetune']['source']:
                finetune_source_rmse = evaluation_results['finetune']['source']['RMSE']
                self.logger.info(f"Finetuned Model (Source): RMSE = {finetune_source_rmse:.4f}")
            
            if evaluation_results['finetune']['target']:
                finetune_target_rmse = evaluation_results['finetune']['target']['RMSE']
                self.logger.info(f"Finetuned Model (Target): RMSE = {finetune_target_rmse:.4f}")
            
            self.logger.info(f"Results saved in: {self.exp_dir}")
            
            return {
                'pretrain': pretrain_results,
                'finetune': finetune_results,
                'evaluation': evaluation_results,
                'experiment_directory': self.exp_dir
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


def main():
    """
    メイン実行関数
    """
    # 設定ファイルのパス
    current_dir = Path(__file__).parent
    config_path = current_dir / 'config' / 'config.yaml'
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    print("="*60)
    print("Fine-Tuning Pipeline for Domain Adaptation (ver1.00)")
    print("FeCu (Source) → FeFe (Target)")
    print("="*60)
    
    try:
        # パイプライン実行
        pipeline = FineTuningPipeline(str(config_path))
        results = pipeline.run_full_pipeline()
        
        print("\nPipeline execution completed successfully!")
        print(f"Results saved in: {results['experiment_directory']}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()