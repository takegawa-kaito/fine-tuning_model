# Fine-Tuning Pipeline for Domain Adaptation (ver1.00)

PyTorchベースのドメイン適応ファインチューニングパイプライン

## 概要

このプロジェクトは、FeCu（ソースドメイン）からFeFe（ターゲットドメイン）への回帰モデルのドメイン適応を行うファインチューニングシステムです。

### 主な機能

- **事前学習**: FeCuデータでのニューラルネットワーク学習
- **ファインチューニング**: FeFeデータでの事後学習
- **評価・可視化**: 詳細な性能分析と可視化
- **実験管理**: 設定ベースの実験管理

## プロジェクト構造

```
simple-FT_ver1.00/
├── main.py                 # メイン実行スクリプト
├── config/
│   └── config.yaml        # 設定ファイル
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # データ読み込み・前処理
│   ├── models.py          # PyTorchモデル定義
│   ├── trainer.py         # 学習・トレーニング管理
│   └── evaluator.py       # 評価・可視化
├── data/                  # 出力データ（自動生成）
├── models/                # 保存モデル（自動生成）
├── requirements.txt       # 依存パッケージ
└── README.md             # このファイル
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 設定ファイルの確認

`config/config.yaml`でデータパスや学習パラメータを確認・調整してください。

### 3. データファイルの配置

以下のCSVファイルが正しい場所にあることを確認：

- FeCuデータ（ソース）:
  - `fecu_JF_L4mm_20250120_start_end_150_300_0_rev2_removebaddata.csv`
  - `fecu_JF_L4mm_add_20250130_start_end_625_500_0_rev2_removebaddata.csv`

- FeFeデータ（ターゲット）:
  - `fefe_JF_L4mm_20250401_start_end_3000_1500_0_rev1.csv`
  - `fefe_JF_L4mm_add_20250404_start_end_625_500_0_rev1.csv`

## 実行方法

### 基本実行

```bash
python main.py
```

### 設定のカスタマイズ

`config/config.yaml`を編集して以下をカスタマイズ可能：

- **データパス**: CSVファイルの場所
- **モデルアーキテクチャ**: 層数、ニューロン数など
- **学習パラメータ**: エポック数、学習率、バッチサイズなど
- **評価設定**: メトリクス、可視化オプション

## 出力結果

実行後、`data/experiments/[実験名]_[タイムスタンプ]/`に以下が生成されます：

### ディレクトリ構造
```
experiments/[実験名]_[タイムスタンプ]/
├── models/
│   ├── pretrained_model.pth      # 事前学習モデル
│   └── finetuned_model.pth       # ファインチューニング済みモデル
├── plots/
│   ├── pretrain_curves.png       # 事前学習の学習曲線
│   ├── finetune_curves.png       # ファインチューニングの学習曲線
│   ├── source_predictions.png    # ソースドメインの予測結果
│   ├── target_predictions.png    # ターゲットドメインの予測結果
│   ├── source_residuals.png      # ソースドメインの残差分析
│   ├── target_residuals.png      # ターゲットドメインの残差分析
│   ├── domain_analysis.png       # ドメイン間比較分析
│   ├── feature_importance.png    # 特徴量重要度
│   └── model_comparison.png      # モデル性能比較
├── results/
│   ├── final_results.json        # 詳細結果（JSON）
│   └── results_summary.csv       # 結果サマリー（CSV）
├── logs/
│   └── training.log              # 学習ログ
└── config.yaml                   # 使用した設定ファイル
```

## モデルアーキテクチャ

### SimpleRegressor

- **入力**: Power, Velocity (2次元)
- **出力**: fz (1次元)
- **構造**: 多層パーセプトロン
- **特徴**: 
  - バッチ正規化
  - ドロップアウト
  - 設定可能なアクティベーション関数

### デフォルト設定

```yaml
model:
  architecture:
    input_dim: 2
    hidden_layers: [64, 32, 16]
    output_dim: 1
    activation: "ReLU"
    dropout_rate: 0.1
    batch_norm: true
```

## 学習プロセス

### 1. 事前学習（Pretrain）
- **データ**: FeCu（ソースドメイン）
- **目的**: 基礎的な回帰能力の獲得
- **設定**: より長いエポック数、高い学習率

### 2. ファインチューニング（Finetune）
- **データ**: FeFe（ターゲットドメイン）
- **目的**: ターゲットドメインへの適応
- **設定**: 短いエポック数、低い学習率
- **実験条件**: learning_data_num = 64, 128, 256でRMSE精度保持を検証
- **オプション**: 層の凍結設定

## 評価メトリクス

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: 決定係数
- **MAPE**: Mean Absolute Percentage Error
- **その他**: 残差統計、ドメインギャップ分析

## カスタマイズ例

### 学習率の調整

```yaml
pretrain:
  learning_rate: 0.001

finetune:
  learning_rate: 0.0001  # より小さく設定
```

### モデル構造の変更

```yaml
model:
  architecture:
    hidden_layers: [128, 64, 32]  # より大きなネットワーク
    dropout_rate: 0.2             # より強いドロップアウト
```

### 層の凍結設定

```yaml
finetune:
  freeze_layers:
    enable: true
    freeze_until_layer: 2  # 最初の2層を凍結
```

## トラブルシューティング

### よくある問題

1. **CUDA out of memory**: バッチサイズを小さくする
2. **学習が収束しない**: 学習率を調整する
3. **過学習**: ドロップアウト率を上げる、早期停止を有効にする

### ログの確認

実験ディレクトリ内の`logs/training.log`で詳細な学習経過を確認できます。

## ライセンス

内部使用限定

## 更新履歴

- v1.0.0: 初期リリース
  - FeCu → FeFe ドメイン適応
  - PyTorchベースのファインチューニング
  - 詳細な可視化・分析機能