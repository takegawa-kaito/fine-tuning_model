import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging
import os
import json
import time
from datetime import datetime


class EarlyStopping:
    """
    早期停止のクラス
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False):
        """
        Args:
            patience (int): 改善が見られない最大エポック数
            min_delta (float): 改善とみなす最小の変化量
            verbose (bool): ログ出力するか
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        早期停止の判定
        
        Args:
            val_loss (float): 検証損失
            
        Returns:
            bool: 早期停止するかどうか
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class MetricsCalculator:
    """
    評価メトリクス計算クラス
    """
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        各種評価メトリクスを計算
        
        Args:
            y_true (np.ndarray): 正解値
            y_pred (np.ndarray): 予測値
            
        Returns:
            Dict[str, float]: 評価メトリクス
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }


class Trainer:
    """
    モデル学習のトレーナークラス
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict, device: torch.device):
        """
        Args:
            model (torch.nn.Module): 学習対象モデル
            config (Dict): 設定辞書
            device (torch.device): 使用デバイス
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # モデルをデバイスに移動
        self.model.to(self.device)
        
        # 学習履歴
        self.train_history = {'loss': [], 'val_loss': []}
        self.metrics_history = {}
        
        # 最良モデルの保存用
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
    def setup_optimizer(self, learning_rate: float, optimizer_name: str = 'Adam') -> torch.optim.Optimizer:
        """
        オプティマイザーの設定
        
        Args:
            learning_rate (float): 学習率
            optimizer_name (str): オプティマイザー名
            
        Returns:
            torch.optim.Optimizer: オプティマイザー
        """
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: torch.optim.Optimizer, 
                       scheduler_config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        学習率スケジューラーの設定
        
        Args:
            optimizer (torch.optim.Optimizer): オプティマイザー
            scheduler_config (Dict): スケジューラー設定
            
        Returns:
            Optional[torch.optim.lr_scheduler._LRScheduler]: スケジューラー
        """
        if not scheduler_config.get('use', False):
            return None
        
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'StepLR':
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 50),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 100)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 10),
                factor=scheduler_config.get('factor', 0.5)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   loss_fn: torch.nn.Module) -> float:
        """
        1エポックの学習
        
        Args:
            train_loader (DataLoader): 学習データローダー
            optimizer (Optimizer): オプティマイザー
            loss_fn (Module): 損失関数
            
        Returns:
            float: 学習損失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 勾配をゼロに
            optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(batch_X)
            loss = loss_fn(outputs, batch_y)
            
            # 逆伝播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader, 
                      loss_fn: torch.nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        1エポックの検証
        
        Args:
            val_loader (DataLoader): 検証データローダー
            loss_fn (Module): 損失関数
            
        Returns:
            Tuple[float, Dict[str, float]]: (検証損失, メトリクス)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 予測値と正解値を保存
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        val_loss = total_loss / num_batches
        
        # メトリクス計算
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        metrics = MetricsCalculator.calculate_metrics(targets, predictions)
        
        return val_loss, metrics
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader,
             config_key: str = 'pretrain') -> Dict:
        """
        モデルの学習を実行
        
        Args:
            train_loader (DataLoader): 学習データローダー
            val_loader (DataLoader): 検証データローダー
            config_key (str): 設定のキー ('pretrain' or 'finetune')
            
        Returns:
            Dict: 学習結果
        """
        train_config = self.config[config_key]
        
        # オプティマイザーとスケジューラーの設定
        optimizer = self.setup_optimizer(
            train_config['learning_rate'], 
            train_config['optimizer']
        )
        scheduler = self.setup_scheduler(optimizer, train_config['scheduler'])
        
        # 損失関数の設定
        from .models import get_loss_function
        loss_fn = get_loss_function(train_config['loss_function'])
        
        # 早期停止の設定
        early_stopping = EarlyStopping(
            patience=train_config['early_stopping']['patience'],
            min_delta=train_config['early_stopping']['min_delta'],
            verbose=True
        )
        
        # 学習ループ
        self.logger.info(f"Starting {config_key} training...")
        start_time = time.time()
        
        for epoch in range(train_config['epochs']):
            # 学習
            train_loss = self.train_epoch(train_loader, optimizer, loss_fn)
            
            # 検証
            val_loss, val_metrics = self.validate_epoch(val_loader, loss_fn)
            
            # 履歴に記録
            self.train_history['loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            for metric_name, metric_value in val_metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                self.metrics_history[metric_name].append(metric_value)
            
            # スケジューラーの更新
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 最良モデルの保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # ログ出力
            if epoch % 10 == 0 or epoch == train_config['epochs'] - 1:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch+1}/{train_config['epochs']}: "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"RMSE: {val_metrics['RMSE']:.4f}, R2: {val_metrics['R2']:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            
            # 早期停止の判定
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 最良モデルの読み込み
        if train_config['save_best_model'] and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model from training")
        
        training_time = time.time() - start_time
        
        # 結果のまとめ
        final_val_loss, final_metrics = self.validate_epoch(val_loader, loss_fn)
        
        results = {
            'final_val_loss': final_val_loss,
            'final_metrics': final_metrics,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'total_epochs': epoch + 1,
            'train_history': self.train_history.copy(),
            'metrics_history': self.metrics_history.copy()
        }
        
        self.logger.info(f"{config_key} training completed in {training_time:.2f}s")
        self.logger.info(f"Final validation metrics: {final_metrics}")
        
        return results
    
    def predict(self, data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測を実行
        
        Args:
            data_loader (DataLoader): データローダー
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (予測値, 正解値)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                
                outputs = self.model(batch_X)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        return predictions, targets
    
    def save_model(self, save_path: str, additional_info: Dict = None):
        """
        モデルの保存
        
        Args:
            save_path (str): 保存先パス
            additional_info (Dict): 追加情報
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config['model'],
            'train_history': self.train_history,
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> Dict:
        """
        モデルの読み込み
        
        Args:
            load_path (str): 読み込みパス
            
        Returns:
            Dict: 読み込んだ情報
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        if 'metrics_history' in checkpoint:
            self.metrics_history = checkpoint['metrics_history']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Model loaded from {load_path}")
        return checkpoint
    
    def plot_training_curves(self, save_path: str = None):
        """
        学習曲線をプロット
        
        Args:
            save_path (str): 保存先パス（Noneの場合は表示のみ）
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 損失曲線
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # RMSE曲線
        if 'RMSE' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['RMSE'], label='RMSE', color='orange')
            axes[0, 1].set_title('RMSE Curve')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # R2曲線
        if 'R2' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['R2'], label='R2 Score', color='green')
            axes[1, 0].set_title('R2 Score Curve')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R2 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # MAE曲線
        if 'MAE' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['MAE'], label='MAE', color='red')
            axes[1, 1].set_title('MAE Curve')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()