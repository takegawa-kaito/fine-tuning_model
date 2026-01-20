import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging


class SimpleRegressor(nn.Module):
    """
    エンコーダー・リグレッサー構成の回帰モデル
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config (Dict): モデル設定辞書
        """
        super(SimpleRegressor, self).__init__()
        
        self.config = config['model']['architecture']
        self.logger = logging.getLogger(__name__)
        
        # 設定から次元を取得
        X_num = self.config['input_dim']
        hidden_size = self.config.get('hidden_size')
        
        # エンコーダー（特徴抽出器）
        self.encoder = nn.Sequential(
            nn.Linear(X_num, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        
        # 回帰器（メインタスク）
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        
        # ドロップアウト（既存との互換性）
        self.dropout = nn.Dropout(self.config.get('dropout_rate', 0.1))
        
        self.logger.info(f"Model architecture: {self}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x (torch.Tensor): 入力テンソル [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 出力テンソル [batch_size, 1]
        """
        # エンコーダーで特徴抽出
        encoded = self.encoder(x)
        
        # リグレッサーで回帰
        output = self.regressor(encoded)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        エンコーダーで特徴量を抽出
        
        Args:
            x (torch.Tensor): 入力テンソル [batch_size, input_dim]
            
        Returns:
            torch.Tensor: エンコードされた特徴量 [batch_size, hidden_size]
        """
        return self.encoder(x)
        """
        指定した層の特徴量を取得（可視化・解析用）
        
        Args:
            x (torch.Tensor): 入力テンソル
            layer_idx (int): 特徴量を取得する層のインデックス（-1で最終隠れ層）
            
        Returns:
            torch.Tensor: 特徴量テンソル
        """
        layer_count = 0
        target_layer = len(self.config['hidden_layers']) + layer_idx if layer_idx < 0 else layer_idx
        
        layer_output_idx = 0
        
        for i in range(len(self.config['hidden_layers'])):
            # 線形変換
            x = self.layers[layer_output_idx](x)
            layer_output_idx += 1
            
            # バッチ正規化
            if self.config['batch_norm']:
                x = self.layers[layer_output_idx](x)
                layer_output_idx += 1
            
            # アクティベーション
            x = self.activation(x)
            
            # 指定層に到達した場合は特徴量を返す
            if layer_count == target_layer:
                return x
            
            # ドロップアウト
            x = self.dropout(x)
            layer_count += 1
        
        return x
    
    def freeze_layers(self, freeze_until_layer: int):
        """
        指定した層まで重みを凍結
        
        Args:
            freeze_until_layer (int): 凍結する層数
        """
        if freeze_until_layer <= 0:
            return
        
        layer_count = 0
        
        for i, layer in enumerate(self.layers):
            if layer_count >= freeze_until_layer:
                break
            
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = False
                layer_count += 1
        
        self.logger.info(f"Froze {freeze_until_layer} layers")
    
    def unfreeze_all_layers(self):
        """
        全ての層の凍結を解除
        """
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.logger.info("Unfroze all layers")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        パラメータ数を取得
        
        Returns:
            Dict[str, int]: パラメータ数の辞書
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


class ModelFactory:
    """
    モデル生成のファクトリクラス
    """
    
    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """
        設定に基づいてモデルを作成
        
        Args:
            config (Dict): 設定辞書
            
        Returns:
            nn.Module: 作成されたモデル
        """
        model_name = config['model']['name']
        
        if model_name == "SimpleRegressor":
            return SimpleRegressor(config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")


# 損失関数の定義
class CustomLoss:
    """
    カスタム損失関数のコレクション
    """
    
    @staticmethod
    def mse_loss():
        """平均二乗誤差損失"""
        return nn.MSELoss()
    
    @staticmethod
    def mae_loss():
        """平均絶対誤差損失"""
        return nn.L1Loss()
    
    @staticmethod
    def huber_loss(delta: float = 1.0):
        """Huber損失"""
        return nn.HuberLoss(delta=delta)
    
    @staticmethod
    def smooth_l1_loss():
        """Smooth L1損失"""
        return nn.SmoothL1Loss()


def get_loss_function(loss_name: str) -> nn.Module:
    """
    損失関数を取得
    
    Args:
        loss_name (str): 損失関数名
        
    Returns:
        nn.Module: 損失関数
    """
    loss_functions = {
        'MSELoss': CustomLoss.mse_loss(),
        'MAELoss': CustomLoss.mae_loss(),
        'HuberLoss': CustomLoss.huber_loss(),
        'SmoothL1Loss': CustomLoss.smooth_l1_loss()
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name]


def init_weights(module: nn.Module):
    """
    重みの初期化
    
    Args:
        module (nn.Module): 初期化するモジュール
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform初期化
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        # バッチ正規化の初期化
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)