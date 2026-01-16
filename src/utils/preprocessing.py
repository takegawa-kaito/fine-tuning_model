import pandas as pd
import os
from typing import Dict, List, Union
import yaml


def preprocess_data(config: Dict) -> None:
    """
    CSVファイルから指定された列を抽出し、前処理済みファイルとして保存する関数
    
    Args:
        config (Dict): 設定辞書
            - raw_filenames (str or List[str]): rawファイル名（複数選択可能）
            - preprocessed_filename (str): preprocessedファイル名
            - input_path (str): inputフォルダまでの絶対パス
            - output_path (str, optional): 出力フォルダの絶対パス（省略時はinput_path/preprocessed）
            - columns (List[str]): csvから抽出する列名のリスト
    """
    
    # 設定値の取得
    raw_filenames = config.get('raw_filenames')
    preprocessed_filename = config.get('preprocessed_filename')
    input_path = config.get('input_path')
    output_path = config.get('output_path')  # 追加：出力パス
    columns = config.get('columns')
    
    # 必要なパラメータの検証
    if not raw_filenames:
        raise ValueError("raw_filenames is required")
    if not preprocessed_filename:
        raise ValueError("preprocessed_filename is required")
    if not input_path:
        raise ValueError("input_path is required")
    if not columns:
        raise ValueError("columns is required")
    
    # raw_filenamesが文字列の場合はリストに変換
    if isinstance(raw_filenames, str):
        raw_filenames = [raw_filenames]
    
    # rawフォルダのパスを構築
    raw_folder = input_path
    
    # 出力パスの設定（指定されていない場合は従来通りinput_path/preprocessedを使用）
    if output_path:
        preprocessed_folder = output_path
    else:
        preprocessed_folder = os.path.join(input_path, 'preprocessed')
    
    # preprocessedフォルダが存在しない場合は作成
    os.makedirs(preprocessed_folder, exist_ok=True)
    
    # データフレームのリストを初期化
    dataframes = []
    
    # 各rawファイルを処理
    for filename in raw_filenames:
        raw_file_path = os.path.join(raw_folder, filename)
        
        # ファイルの存在確認
        if not os.path.exists(raw_file_path):
            raise FileNotFoundError(f"Raw file not found: {raw_file_path}")
        
        # CSVファイルの読み込み
        print(f"Processing file: {filename}")
        df = pd.read_csv(raw_file_path)
        
        # 指定された列が存在するか確認
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {filename}: {missing_columns}")
        
        # 指定された列のみを抽出
        df_extracted = df[columns]
        dataframes.append(df_extracted)
        print(f"Extracted {len(df_extracted)} rows from {filename}")
    
    # データフレームを行方向に結合
    if len(dataframes) == 1:
        combined_df = dataframes[0]
    else:
        combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
        print(f"Combined {len(dataframes)} files into {len(combined_df)} total rows")
    
    # preprocessedファイルとして保存
    output_path = os.path.join(preprocessed_folder, preprocessed_filename)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to: {output_path}")
    print(f"Output shape: {combined_df.shape}")
    
    return combined_df


def load_config_from_yaml(yaml_path: str) -> Dict:
    """
    YAMLファイルから設定を読み込む
    
    Args:
        yaml_path (str): YAMLファイルのパス
        
    Returns:
        Dict: 設定辞書
    """
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def run_preprocessing_from_yaml(yaml_path: str) -> None:
    """
    YAMLファイルを読み込んで前処理を実行
    
    Args:
        yaml_path (str): YAMLファイルのパス
    """
    config = load_config_from_yaml(yaml_path)
    preprocess_data(config)


if __name__ == "__main__":
    # テスト実行用
    import sys
    
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
        run_preprocessing_from_yaml(yaml_path)
    else:
        print("Usage: python preprocessing.py <yaml_config_file>")
