import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import logging
import os


class Evaluator:
    """
    モデル評価・可視化クラス
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): 設定辞書
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # プロット設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        詳細な評価メトリクスを計算
        
        Args:
            y_true (np.ndarray): 正解値
            y_pred (np.ndarray): 予測値
            
        Returns:
            Dict[str, float]: 評価メトリクス
        """
        # 基本メトリクス
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 追加メトリクス
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        max_error = np.max(np.abs(residuals))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(residuals / (y_true + 1e-8))) * 100
        
        # Explained Variance Score
        explained_var = 1 - (np.var(residuals) / np.var(y_true))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Mean_Residual': mean_residual,
            'Std_Residual': std_residual,
            'Max_Error': max_error,
            'MAPE': mape,
            'Explained_Variance': explained_var
        }
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual", 
                        save_path: str = None) -> None:
        """
        予測値 vs 実測値のプロット
        
        Args:
            y_true (np.ndarray): 正解値
            y_pred (np.ndarray): 予測値
            title (str): プロットタイトル
            save_path (str): 保存先パス
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 散布図
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # 理想直線
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predictions vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R2スコアを表示
        r2 = r2_score(y_true, y_pred)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 残差プロット
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
                self.logger.info(f"Predictions plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save predictions plot: {e}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               title: str = "Residuals Analysis", 
                               save_path: str = None) -> None:
        """
        残差の詳細分析プロット
        
        Args:
            y_true (np.ndarray): 正解値
            y_pred (np.ndarray): 予測値
            title (str): プロットタイトル
            save_path (str): 保存先パス
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 残差ヒストグラム
        axes[0, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Residuals Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均と標準偏差を表示
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[0, 0].axvline(mean_res, color='red', linestyle='--', 
                          label=f'Mean: {mean_res:.3f}')
        axes[0, 0].axvline(mean_res + std_res, color='orange', linestyle='--', 
                          label=f'Mean + Std: {mean_res + std_res:.3f}')
        axes[0, 0].axvline(mean_res - std_res, color='orange', linestyle='--', 
                          label=f'Mean - Std: {mean_res - std_res:.3f}')
        axes[0, 0].legend()
        
        # Q-Qプロット
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差 vs 予測値
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差の絶対値 vs 予測値
        axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('|Residuals|')
        axes[1, 1].set_title('Absolute Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
                self.logger.info(f"Residuals analysis plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save residuals plot: {e}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               title: str = "Feature Analysis", 
                               save_path: str = None) -> None:
        """
        特徴量の重要度分析（線形層の重みを可視化）
        
        Args:
            model: 学習済みモデル
            feature_names (List[str]): 特徴量名のリスト
            title (str): プロットタイトル
            save_path (str): 保存先パス
        """
        # 第1層の重みを取得
        first_layer = None
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                first_layer = layer
                break
        
        if first_layer is None:
            self.logger.warning("No linear layer found for feature importance analysis")
            return
        
        weights = first_layer.weight.data.cpu().numpy()  # [output_dim, input_dim]
        
        # 各特徴量の重みの絶対値の平均を計算
        importance = np.mean(np.abs(weights), axis=0)
        
        # プロット
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 棒グラフ
        bars = axes[0].bar(feature_names, importance)
        axes[0].set_title('Feature Importance (Avg. Absolute Weight)')
        axes[0].set_ylabel('Importance Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 色分け
        colors = plt.cm.viridis(importance / importance.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 重みのヒートマップ
        im = axes[1].imshow(weights, aspect='auto', cmap='RdBu_r')
        axes[1].set_title('Weight Matrix Heatmap')
        axes[1].set_xlabel('Input Features')
        axes[1].set_ylabel('Hidden Units')
        axes[1].set_xticks(range(len(feature_names)))
        axes[1].set_xticklabels(feature_names, rotation=45)
        
        # カラーバー
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save feature importance plot: {e}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models(
        self,
        results: Dict[str, Dict],
        title: str = "Model Comparison",
        save_path: str = None
    ) -> None:
        """
        複数モデルの性能比較

        Args:
            results (Dict[str, Dict]): モデル名をキーとした結果辞書
            title (str): プロットタイトル
            save_path (str): 保存先パス
        """
        model_names = list(results.keys())
        metrics = ['RMSE', 'MAE', 'R2']

        # メトリクス値を抽出
        metric_values = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                if metric in results[model_name]:
                    metric_values[metric].append(results[model_name][metric])
                else:
                    metric_values[metric].append(0)

        # プロット
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(metrics):
            bars = axes[i].bar(model_names, metric_values[metric])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)

            # 値をバーの上に表示
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom'
                )

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
                self.logger.info(f"Model comparison plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save model comparison plot: {e}")
        else:
            plt.show()

        plt.close()

    def plot_domain_analysis(self, source_results: Dict, target_results: Dict,
                            source_data: Tuple[np.ndarray, np.ndarray],
                            target_data: Tuple[np.ndarray, np.ndarray],
                            save_path: str = None) -> None:
        """
        ドメイン間の分析プロット
        Args:
            source_results (Dict): ソースドメインの結果
            target_results (Dict): ターゲットドメインの結果
            source_data (Tuple): ソースドメインのデータ
            target_data (Tuple): ターゲットドメインのデータ
            save_path (str): 保存先パス
        """
        source_X, source_y = source_data
        target_X, target_y = target_data

        # データクリーニングを実行
        source_X, source_y = self._clean_data_for_analysis(source_X, source_y)
        target_X, target_y = self._clean_data_for_analysis(target_X, target_y)
        
        # データ型を確実に数値型に変換
        source_X = np.asarray(source_X, dtype=np.float64)
        source_y = np.asarray(source_y, dtype=np.float64).flatten()
        target_X = np.asarray(target_X, dtype=np.float64)
        target_y = np.asarray(target_y, dtype=np.float64).flatten()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # データ分布の比較
        feature_names = self.config['model_data']['features']
        
        for i, feature in enumerate(feature_names[:2]):  # 最初の2つの特徴量
            # 特徴量分布
            axes[0, i].hist(source_X[:, i], bins=30, alpha=0.6, label='Source (FeCu)', density=True)
            axes[0, i].hist(target_X[:, i], bins=30, alpha=0.6, label='Target (FeFe)', density=True)
            axes[0, i].set_xlabel(feature)
            axes[0, i].set_ylabel('Density')
            axes[0, i].set_title(f'{feature} Distribution')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # 目的変数の分布
        axes[0, 2].hist(source_y, bins=30, alpha=0.6, label='Source (FeCu)', density=True)
        axes[0, 2].hist(target_y, bins=30, alpha=0.6, label='Target (FeFe)', density=True)
        axes[0, 2].set_xlabel('fz')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Target Variable Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 性能比較
        metrics = ['RMSE', 'MAE', 'R2']
        source_metrics = [source_results.get(m, 0) for m in metrics]
        target_metrics = [target_results.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, source_metrics, width, label='Source Domain', alpha=0.8)
        axes[1, 0].bar(x + width/2, target_metrics, width, label='Target Domain', alpha=0.8)
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Values')
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # ドメイン間の相関
        if len(feature_names) >= 2:
            axes[1, 1].scatter(source_X[:, 0], source_X[:, 1], alpha=0.6, label='Source (FeCu)', s=30)
            axes[1, 1].scatter(target_X[:, 0], target_X[:, 1], alpha=0.6, label='Target (FeFe)', s=30)
            axes[1, 1].set_xlabel(feature_names[0])
            axes[1, 1].set_ylabel(feature_names[1])
            axes[1, 1].set_title('Feature Space Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # ドメインギャップの統計
        stats_text = []
        for i, feature in enumerate(feature_names):
            source_mean = np.mean(source_X[:, i])
            target_mean = np.mean(target_X[:, i])
            gap = abs(target_mean - source_mean)
            stats_text.append(f'{feature}: {gap:.2f}')
        
        target_mean_y = np.mean(target_y)
        source_mean_y = np.mean(source_y)
        y_gap = abs(target_mean_y - source_mean_y)
        stats_text.append(f'fz: {y_gap:.2f}')
        
        axes[1, 2].text(0.1, 0.5, 'Domain Gap (|mean_diff|):\n' + '\n'.join(stats_text),
                       transform=axes[1, 2].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        axes[1, 2].set_title('Domain Statistics')
        axes[1, 2].axis('off')
        
        plt.suptitle('Domain Adaptation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
                self.logger.info(f"Domain analysis plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Could not save domain analysis plot: {e}")
                # 代替保存方法を試行
                try:
                    plt.savefig(save_path.replace('.png', '_backup.png'), format='png', bbox_inches='tight')
                    self.logger.info(f"Domain analysis plot saved to {save_path.replace('.png', '_backup.png')}")
                except Exception as e2:
                    self.logger.error(f"Failed to save domain analysis plot: {e2}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results_summary(self, results: Dict, save_path: str) -> None:
        """
        結果の詳細サマリーをCSVで保存
        
        Args:
            results (Dict): 結果辞書
            save_path (str): 保存先パス
        """
        summary_data = []
        
        for phase, phase_results in results.items():
            if isinstance(phase_results, dict) and 'final_metrics' in phase_results:
                row = {'Phase': phase}
                row.update(phase_results['final_metrics'])
                row['Best_Val_Loss'] = phase_results.get('best_val_loss', 'N/A')
                row['Training_Time'] = phase_results.get('training_time', 'N/A')
                row['Total_Epochs'] = phase_results.get('total_epochs', 'N/A')
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Results summary saved to {save_path}")
        
        return df
    
    def _clean_data_for_analysis(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        分析用データのクリーニング
        
        Args:
            X (np.ndarray): 特徴量データ
            y (np.ndarray): 目的変数データ
            
        Returns:
            tuple: クリーニング済みデータ (X, y)
        """
        def is_numeric_convertible(arr):
            """配列が数値に変換可能かチェック"""
            if arr.dtype == 'object':
                try:
                    mask = np.zeros(len(arr), dtype=bool)
                    for i, val in enumerate(arr):
                        try:
                            float(str(val))
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
            print(f"Warning: {invalid_count} rows with invalid data removed in analysis")
        
        return X[valid_mask], y[valid_mask]