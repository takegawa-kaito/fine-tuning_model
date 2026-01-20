# Grid Search for Fine-Tuning Pipeline

グリッドサーチ機能の使用方法とパラメータ設定のガイドです。

## 概要

このグリッドサーチ機能では、以下のパラメータを系統的に探索できます：
- **オプティマイザー**: Adam, AdamW, SGD
- **学習率**: 事前学習とファインチューニング別設定
- **スケジューラー**: StepLR, CosineAnnealingLR
- **モデルアーキテクチャ**: hidden_size, dropout_rate
- **バッチサイズ**: 事前学習とファインチューニング別設定

## 設定方法

### 1. config.yaml でグリッドサーチを有効化

```yaml
grid_search:
  enable: true  # グリッドサーチを有効化
  max_experiments: 20  # 最大実験数（組み合わせ数制限）
  
  parameters:
    # オプティマイザー比較
    optimizer: ["Adam", "AdamW", "SGD"]
    
    # 学習率グリッド
    learning_rate:
      pretrain: [0.001, 0.005, 0.01]
      finetune: [0.0001, 0.0005, 0.001]
    
    # スケジューラー設定
    scheduler:
      type: ["StepLR", "CosineAnnealingLR"]
      cosine_T_max: [50, 100, 150]  # Cosine用
      step_gamma: [0.5, 0.7, 0.9]   # StepLR用
    
    # モデルアーキテクチャ
    model:
      hidden_size: [128, 256, 512]
      dropout_rate: [0.1, 0.15, 0.2]
    
    # バッチサイズ
    batch_size:
      pretrain: [16, 32, 64]
      finetune: [8, 16, 32]
  
  # 結果保存設定
  save_results: true
  save_best_only: true
  metric_for_best: "finetune_target_rmse"  # 最適化指標
```

### 2. 実行方法

```bash
# グリッドサーチ実行
cd src/simple-FT_ver1.00
python grid_search.py
```

## 推奨設定例

### 軽量実験（5-10実験）

```yaml
grid_search:
  enable: true
  max_experiments: 10
  
  parameters:
    optimizer: ["Adam", "AdamW"]
    learning_rate:
      pretrain: [0.001, 0.005]
      finetune: [0.0001, 0.0005]
    scheduler:
      type: ["StepLR", "CosineAnnealingLR"]
    model:
      hidden_size: [256, 512]
```

### 中規模実験（15-20実験）

```yaml
grid_search:
  enable: true
  max_experiments: 20
  
  parameters:
    optimizer: ["Adam", "AdamW", "SGD"]
    learning_rate:
      pretrain: [0.001, 0.005]
      finetune: [0.0001, 0.0005, 0.001]
    scheduler:
      type: ["StepLR", "CosineAnnealingLR"]
      cosine_T_max: [50, 100]
    model:
      hidden_size: [128, 256, 512]
      dropout_rate: [0.1, 0.15]
```

## 結果の確認

### 生成されるファイル

実行後、以下のファイルが生成されます：

```
data/grid_search_results/grid_search_YYYYMMDD_HHMMSS/
├── grid_search_results.json      # 全実験の詳細結果
├── grid_search_summary.csv       # 成功した実験のサマリー
├── intermediate_results.json     # 中間結果（実行中随時更新）
└── config_exp_*.yaml            # 各実験の設定ファイル
```

### CSV サマリー内容

```csv
experiment_id,pretrain_optimizer,finetune_optimizer,pretrain_lr,finetune_lr,
pretrain_scheduler,finetune_scheduler,hidden_size,dropout_rate,
pretrain_source_rmse,finetune_source_rmse,finetune_target_rmse,
pretrain_source_r2,finetune_source_r2,finetune_target_r2
```

### 結果分析ログ例

```
=================================================
GRID SEARCH ANALYSIS
=================================================
Best result (lowest finetune_target_rmse):
  Experiment ID: 12
  Target RMSE: 32.4567
  Target R²: 0.7856
  Parameters:
    pretrain_optimizer: AdamW
    finetune_optimizer: AdamW
    pretrain_lr: 0.005
    finetune_lr: 0.0005
    pretrain_scheduler: CosineAnnealingLR
    finetune_scheduler: CosineAnnealingLR
    hidden_size: 512
    dropout_rate: 0.15
```

## 最適化のポイント

### 1. 学習率組み合わせ

- **事前学習**: 0.001-0.01 範囲で探索
- **ファインチューニング**: 事前学習の1/10～1/2程度
- **SGD**: より大きな学習率が必要（0.01-0.1）

### 2. オプティマイザー比較

- **Adam**: 安定した収束、デフォルト選択
- **AdamW**: Weight Decayが正しく機能、過学習抑制
- **SGD**: モメンタム使用、最終性能が高い場合がある

### 3. スケジューラー設定

- **StepLR**: 分かりやすく安定、gamma=0.5-0.7推奨
- **CosineAnnealingLR**: 滑らかな学習率変化、T_max=epochs/2推奨

### 4. アーキテクチャ調整

- **hidden_size**: 128→256→512で段階的テスト
- **dropout_rate**: 0.1-0.2範囲、大きすぎると学習阻害

## トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```yaml
   batch_size:
     pretrain: [16, 32]  # 64は除外
     finetune: [8, 16]   # 32は除外
   ```

2. **実行時間が長い**
   ```yaml
   max_experiments: 10  # 20から削減
   ```

3. **収束しない組み合わせ**
   - SGD + 高dropout_rate
   - 大きすぎる学習率
   - 小さすぎるhidden_size

### ログの確認

実行中のログで各実験の進行状況を確認：
```
Running experiment 5: gridsearch_optimizer_AdamW_pretrain_lr_0.005...
Experiment 5 completed successfully
Target RMSE: 34.2156
```

## 単一実験実行

グリッドサーチを無効化して通常実行：

```yaml
grid_search:
  enable: false  # 無効化
```

```bash
python main.py  # 通常のパイプライン実行
```

---

*Grid Search Documentation*  
*Version: 1.0*  
*Updated: January 20, 2026*