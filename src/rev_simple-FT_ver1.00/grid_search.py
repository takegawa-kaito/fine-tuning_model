import os
import sys
import yaml
import json
import itertools
import logging
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FineTuningPipeline


class GridSearchManager:
    """
    グリッドサーチ管理クラス
    """
    
    def __init__(self, base_config_path: str):
        """
        Args:
            base_config_path (str): ベース設定ファイルのパス
        """
        self.base_config_path = base_config_path
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        self.grid_config = self.base_config.get('grid_search', {})
        self.results = []
        
        # ログ設定
        self.logger = self._setup_logger()
        
        # 結果保存ディレクトリ
        self.results_dir = os.path.join(
            self.base_config['data_paths']['output_path'], 
            'grid_search_results',
            f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('GridSearch')
        logger.setLevel(logging.INFO)
        
        # コンソールハンドラー
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_parameter_combinations(self) -> List[Dict]:
        """
        パラメータの組み合わせを生成
        
        Returns:
            List[Dict]: パラメータ組み合わせのリスト
        """
        if not self.grid_config.get('enable', False):
            self.logger.info("Grid search is disabled")
            return [{}]
        
        parameters = self.grid_config.get('parameters', {})
        combinations = []
        
        # 各パラメータの組み合わせを生成
        param_lists = {}
        
        # オプティマイザー
        if 'optimizer' in parameters:
            param_lists['optimizer'] = parameters['optimizer']
        
        # 学習率（事前学習・ファインチューニング）
        if 'learning_rate' in parameters:
            lr_params = parameters['learning_rate']
            param_lists['pretrain_lr'] = lr_params.get('pretrain', [0.001])
            param_lists['finetune_lr'] = lr_params.get('finetune', [0.0001])
        
        # スケジューラー
        if 'scheduler' in parameters:
            scheduler_params = parameters['scheduler']
            param_lists['scheduler_type'] = scheduler_params.get('type', ['StepLR'])
            param_lists['cosine_T_max'] = scheduler_params.get('cosine_T_max', [100])
            param_lists['step_gamma'] = scheduler_params.get('step_gamma', [0.7])
        
        # モデルアーキテクチャ
        if 'model' in parameters:
            model_params = parameters['model']
            param_lists['hidden_size'] = model_params.get('hidden_size', [256])
            param_lists['dropout_rate'] = model_params.get('dropout_rate', [0.15])
        
        # バッチサイズ
        if 'batch_size' in parameters:
            batch_params = parameters['batch_size']
            param_lists['pretrain_batch'] = batch_params.get('pretrain', [32])
            param_lists['finetune_batch'] = batch_params.get('finetune', [16])
        
        # 組み合わせ生成
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # 最大実験数制限
        max_experiments = self.grid_config.get('max_experiments', 20)
        if len(all_combinations) > max_experiments:
            # ランダムサンプリング
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(all_combinations), 
                size=max_experiments, 
                replace=False
            )
            all_combinations = [all_combinations[i] for i in selected_indices]
            self.logger.warning(f"Too many combinations ({len(list(itertools.product(*param_values)))}). "
                              f"Randomly selected {max_experiments} combinations.")
        
        # 辞書形式に変換
        for combination in all_combinations:
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        self.logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def create_config_for_combination(self, base_config: Dict, params: Dict) -> Dict:
        """
        パラメータ組み合わせ用の設定を作成
        
        Args:
            base_config (Dict): ベース設定
            params (Dict): パラメータ組み合わせ
            
        Returns:
            Dict: 更新された設定
        """
        config = deepcopy(base_config)
        
        # オプティマイザー設定
        if 'optimizer' in params:
            config['pretrain']['optimizer'] = params['optimizer']
            config['finetune']['optimizer'] = params['optimizer']
        
        # 学習率設定
        if 'pretrain_lr' in params:
            config['pretrain']['learning_rate'] = params['pretrain_lr']
        if 'finetune_lr' in params:
            config['finetune']['learning_rate'] = params['finetune_lr']
        
        # スケジューラー設定
        if 'scheduler_type' in params:
            config['pretrain']['scheduler']['type'] = params['scheduler_type']
            config['finetune']['scheduler']['type'] = params['scheduler_type']
            
            # CosineAnnealingLR用パラメータ
            if params['scheduler_type'] == 'CosineAnnealingLR':
                if 'cosine_T_max' in params:
                    config['pretrain']['scheduler']['T_max'] = params['cosine_T_max']
                    config['finetune']['scheduler']['T_max'] = params['cosine_T_max']
            
            # StepLR用パラメータ
            elif params['scheduler_type'] == 'StepLR':
                if 'step_gamma' in params:
                    config['pretrain']['scheduler']['gamma'] = params['step_gamma']
                    config['finetune']['scheduler']['gamma'] = params['step_gamma']
        
        # モデルアーキテクチャ設定
        if 'hidden_size' in params:
            config['model']['architecture']['hidden_size'] = params['hidden_size']
        if 'dropout_rate' in params:
            config['model']['architecture']['dropout_rate'] = params['dropout_rate']
        
        # バッチサイズ設定
        if 'pretrain_batch' in params:
            config['pretrain']['batch_size'] = params['pretrain_batch']
        if 'finetune_batch' in params:
            config['finetune']['batch_size'] = params['finetune_batch']
        
        # 実験名に組み合わせ情報を追加
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        config['experiment']['name'] = f"gridsearch_{param_str}"
        config['experiment']['description'] = f"Grid search experiment: {param_str}"
        
        return config
    
    def run_single_experiment(self, config: Dict, exp_id: int) -> Dict:
        """
        単一実験の実行
        
        Args:
            config (Dict): 実験設定
            exp_id (int): 実験ID
            
        Returns:
            Dict: 実験結果
        """
        self.logger.info(f"Running experiment {exp_id}: {config['experiment']['name']}")
        
        try:
            # 設定ファイルを一時保存
            temp_config_path = os.path.join(self.results_dir, f"config_exp_{exp_id}.yaml")
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # パイプライン実行
            pipeline = FineTuningPipeline(temp_config_path)
            results = pipeline.run_full_pipeline()
            
            # 結果をまとめる
            experiment_result = {
                'experiment_id': exp_id,
                'parameters': self._extract_key_parameters(config),
                'results': {
                    'pretrain_source_rmse': results['evaluation']['pretrain']['source'].get('RMSE', None),
                    'finetune_source_rmse': results['evaluation']['finetune']['source'].get('RMSE', None),
                    'finetune_target_rmse': results['evaluation']['finetune']['target'].get('RMSE', None),
                    'pretrain_source_r2': results['evaluation']['pretrain']['source'].get('R2', None),
                    'finetune_source_r2': results['evaluation']['finetune']['source'].get('R2', None),
                    'finetune_target_r2': results['evaluation']['finetune']['target'].get('R2', None),
                },
                'experiment_directory': results['experiment_directory'],
                'status': 'success'
            }
            
            self.logger.info(f"Experiment {exp_id} completed successfully")
            self.logger.info(f"Target RMSE: {experiment_result['results']['finetune_target_rmse']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Experiment {exp_id} failed: {str(e)}")
            experiment_result = {
                'experiment_id': exp_id,
                'parameters': self._extract_key_parameters(config),
                'results': {},
                'error': str(e),
                'status': 'failed'
            }
        
        return experiment_result
    
    def _extract_key_parameters(self, config: Dict) -> Dict:
        """設定から主要パラメータを抽出"""
        return {
            'pretrain_optimizer': config['pretrain']['optimizer'],
            'finetune_optimizer': config['finetune']['optimizer'],
            'pretrain_lr': config['pretrain']['learning_rate'],
            'finetune_lr': config['finetune']['learning_rate'],
            'pretrain_scheduler': config['pretrain']['scheduler']['type'],
            'finetune_scheduler': config['finetune']['scheduler']['type'],
            'hidden_size': config['model']['architecture']['hidden_size'],
            'dropout_rate': config['model']['architecture']['dropout_rate'],
            'pretrain_batch_size': config['pretrain']['batch_size'],
            'finetune_batch_size': config['finetune']['batch_size'],
        }
    
    def run_grid_search(self) -> List[Dict]:
        """
        グリッドサーチ実行
        
        Returns:
            List[Dict]: 全実験結果
        """
        self.logger.info("Starting Grid Search")
        
        # パラメータ組み合わせ生成
        combinations = self.generate_parameter_combinations()
        
        if not combinations or combinations == [{}]:
            self.logger.info("No grid search combinations. Running single experiment with base config.")
            pipeline = FineTuningPipeline(self.base_config_path)
            results = pipeline.run_full_pipeline()
            return [{'base_experiment': results}]
        
        # 各組み合わせで実験実行
        all_results = []
        
        # tqdmで進捗表示
        progress_bar = tqdm(
            combinations, 
            desc="Grid Search Progress",
            unit="experiment",
            ncols=100
        )
        
        for i, params in enumerate(progress_bar):
            # 設定作成
            experiment_config = self.create_config_for_combination(self.base_config, params)
            
            # 進捗バーの説明を更新
            progress_bar.set_description(f"Exp {i+1}/{len(combinations)} - {params.get('optimizer', 'Unknown')}")
            
            # 実験実行
            result = self.run_single_experiment(experiment_config, i)
            all_results.append(result)
            
            # 中間結果保存
            self._save_intermediate_results(all_results)
            
            # 成功した実験の性能を進捗バーに表示
            if result.get('status') == 'success' and 'results' in result:
                target_rmse = result['results'].get('finetune_target_rmse', 0)
                progress_bar.set_postfix(RMSE=f"{target_rmse:.2f}")
        
        progress_bar.close()
        
        # 最終結果保存・分析
        self._save_final_results(all_results)
        self._analyze_results(all_results)
        
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """中間結果保存"""
        results_file = os.path.join(self.results_dir, 'intermediate_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_final_results(self, results: List[Dict]):
        """最終結果保存"""
        # JSON保存
        results_file = os.path.join(self.results_dir, 'grid_search_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV保存（成功した実験のみ）
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            df_data = []
            for result in successful_results:
                row = {
                    'experiment_id': result['experiment_id'],
                    **result['parameters'],
                    **result['results']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = os.path.join(self.results_dir, 'grid_search_summary.csv')
            df.to_csv(csv_file, index=False)
            
            self.logger.info(f"Results saved to: {results_file}")
            self.logger.info(f"Summary saved to: {csv_file}")
    
    def _analyze_results(self, results: List[Dict]):
        """結果分析"""
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            self.logger.warning("No successful experiments to analyze")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("GRID SEARCH ANALYSIS")
        self.logger.info("=" * 50)
        
        # 最良結果の特定
        metric = self.grid_config.get('metric_for_best', 'finetune_target_rmse')
        
        if metric in ['finetune_target_rmse', 'finetune_source_rmse', 'pretrain_source_rmse']:
            # RMSE（小さいほど良い）
            best_result = min(successful_results, 
                            key=lambda x: x['results'].get(metric, float('inf')))
            self.logger.info(f"Best result (lowest {metric}):")
        else:
            # R2（大きいほど良い）
            best_result = max(successful_results, 
                            key=lambda x: x['results'].get(metric, -float('inf')))
            self.logger.info(f"Best result (highest {metric}):")
        
        self.logger.info(f"  Experiment ID: {best_result['experiment_id']}")
        self.logger.info(f"  Target RMSE: {best_result['results'].get('finetune_target_rmse', 'N/A'):.4f}")
        self.logger.info(f"  Target R²: {best_result['results'].get('finetune_target_r2', 'N/A'):.4f}")
        self.logger.info(f"  Parameters:")
        for key, value in best_result['parameters'].items():
            self.logger.info(f"    {key}: {value}")
        
        # 統計サマリー
        target_rmses = [r['results'].get('finetune_target_rmse') for r in successful_results 
                       if r['results'].get('finetune_target_rmse') is not None]
        
        # コンソールに綺麗に結果表示
        print("\n" + "="*80)
        print("🔍 GRID SEARCH RESULTS ANALYSIS")
        print("="*80)
        
        if target_rmses:
            print(f"🏆 Best Target RMSE: {np.min(target_rmses):.4f}")
            print(f"📊 Target RMSE Statistics:")
            print(f"   Mean: {np.mean(target_rmses):.4f}")
            print(f"   Std:  {np.std(target_rmses):.4f}")
            print(f"   Min:  {np.min(target_rmses):.4f}")
            print(f"   Max:  {np.max(target_rmses):.4f}")
        
        print(f"\n✅ Successful: {len(successful_results)}/{len(results)} experiments")
        
        # 最良結果の詳細
        if target_rmses:
            best_result = min(successful_results, 
                            key=lambda x: x['results'].get('finetune_target_rmse', float('inf')))
            print(f"\n🎯 Best Configuration:")
            print(f"   Target RMSE: {best_result['results'].get('finetune_target_rmse', 'N/A'):.4f}")
            print(f"   Target R²:   {best_result['results'].get('finetune_target_r2', 'N/A'):.4f}")
            print(f"   Parameters:")
            for key, value in best_result['parameters'].items():
                print(f"     {key}: {value}")
        
        # 詳細CSVファイル作成
        self._create_detailed_csv(successful_results)
        print("="*80)
    
    def _create_detailed_csv(self, successful_results: List[Dict]):
        """詳細なCSV結果ファイルを作成"""
        csv_data = []
        for result in successful_results:
            row = {
                'experiment_id': result['experiment_id'],
                'target_rmse': result['results'].get('finetune_target_rmse'),
                'target_r2': result['results'].get('finetune_target_r2'),
                'target_mae': result['results'].get('finetune_target_mae'),
                'source_rmse': result['results'].get('finetune_source_rmse'),
                'source_r2': result['results'].get('finetune_source_r2'),
                'pretrain_rmse': result['results'].get('pretrain_source_rmse'),
                'pretrain_r2': result['results'].get('pretrain_source_r2'),
                **result['parameters']
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        # RMSEで昇順ソート（良い結果が上に）
        df = df.sort_values('target_rmse', ascending=True)
        
        csv_file = os.path.join(self.results_dir, 'detailed_results.csv')
        df.to_csv(csv_file, index=False, float_format='%.6f')
        
        print(f"📁 Detailed CSV saved: {csv_file}")


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
    
    print("=" * 60)
    print("Grid Search for Fine-Tuning Pipeline")
    print("=" * 60)
    
    try:
        # グリッドサーチマネージャー初期化・実行
        grid_manager = GridSearchManager(str(config_path))
        results = grid_manager.run_grid_search()
        
        print("\nGrid search completed successfully!")
        print(f"Results saved in: {grid_manager.results_dir}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()