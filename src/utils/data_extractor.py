import pandas as pd
import numpy as np
import json
import yaml
import os
from typing import Dict, Tuple, List
import random


def get_feature_columns(df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
    """
    内挿・外挿の対象となる特徴量の列名を取得する
    
    Args:
        df (pd.DataFrame): データフレーム
        exclude_columns (List[str]): 除外する列名のリスト
        
    Returns:
        List[str]: 対象特徴量の列名リスト
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # デフォルトで除外する列
    default_exclude = ['datetime_pos', 'pos']
    
    # Defocusは現在0のみなので除外（値の分散がない場合も除外）
    all_exclude = default_exclude + exclude_columns
    
    feature_columns = []
    for col in df.columns:
        if col not in all_exclude:
            # 値の分散がない列（例：すべて0）は除外
            if df[col].nunique() > 1:
                feature_columns.append(col)
            else:
                print(f"Warning: Column '{col}' has no variance (all values are the same), excluding from interpolation/extrapolation.")
    
    print(f"Target features for interpolation/extrapolation: {feature_columns}")
    return feature_columns


def load_config_from_json(json_path: str) -> Dict:
    """
    JSONファイルから設定を読み込む
    
    Args:
        json_path (str): JSONファイルのパス
        
    Returns:
        Dict: 設定辞書
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config


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


def load_config(config_path: str) -> Dict:
    """
    設定ファイルを読み込む（JSONまたはYAMLを自動判別）
    
    Args:
        config_path (str): 設定ファイルのパス
        
    Returns:
        Dict: 設定辞書
    """
    _, ext = os.path.splitext(config_path.lower())
    
    if ext in ['.yml', '.yaml']:
        return load_config_from_yaml(config_path)
    elif ext == '.json':
        return load_config_from_json(config_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .json, .yml, or .yaml")


def extract_random_data(df: pd.DataFrame, learning_data_num: int, validation_data_num: int, 
                       random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ランダム抽出
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        learning_data_num (int): 学習データ数
        validation_data_num (int): 検証データ数
        random_seed (int): ランダムシード
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (learning_data, target_data)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    total_needed = learning_data_num + validation_data_num
    if len(df) < total_needed:
        raise ValueError(f"データが不足しています。必要: {total_needed}, 利用可能: {len(df)}")
    
    # データをシャッフルして選択
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    learning_data = shuffled_df[:learning_data_num]
    target_data = shuffled_df[learning_data_num:learning_data_num + validation_data_num]
    
    return learning_data, target_data


def extract_interpolation_data(df: pd.DataFrame, learning_data_num: int, validation_data_num: int,
                              random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    内挿データ抽出
    target_dataの全ての特徴量がlearning_dataの範囲内に収まるように抽出
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        learning_data_num (int): 学習データ数
        validation_data_num (int): 検証データ数
        random_seed (int): ランダムシード
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (learning_data, target_data)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if len(df) < learning_data_num + validation_data_num:
        raise ValueError(f"データが不足しています。必要: {learning_data_num + validation_data_num}, 利用可能: {len(df)}")
    
    # 内挿・外挿の対象となる特徴量を取得（分散のない列は自動除外）
    feature_columns = get_feature_columns(df)
    
    # 戦略: 複数回試行して最適なlearning_dataを見つける
    best_learning_data = None
    best_target_data = None
    max_interpolation_candidates = 0
    
    for attempt in range(100):  # 最大100回試行
        # learning_dataをランダム選択
        current_seed = random_seed + attempt
        shuffled_df = df.sample(frac=1, random_state=current_seed).reset_index(drop=True)
        potential_learning_data = shuffled_df[:learning_data_num]
        
        # learning_dataの各特徴量の範囲を計算
        feature_ranges = {}
        for col in feature_columns:
            feature_ranges[col] = {
                'min': potential_learning_data[col].min(),
                'max': potential_learning_data[col].max()
            }
        
        # 残りのデータから内挿条件を満たすデータを探す
        remaining_data = shuffled_df[learning_data_num:]
        interpolation_candidates = remaining_data.copy()
        
        # 各特徴量が範囲内にあるかチェック
        for col in feature_columns:
            min_val = feature_ranges[col]['min']
            max_val = feature_ranges[col]['max']
            interpolation_candidates = interpolation_candidates[
                (interpolation_candidates[col] >= min_val) & 
                (interpolation_candidates[col] <= max_val)
            ]
        
        # 十分な内挿データがある場合
        if len(interpolation_candidates) >= validation_data_num:
            if len(interpolation_candidates) > max_interpolation_candidates:
                max_interpolation_candidates = len(interpolation_candidates)
                best_learning_data = potential_learning_data
                best_target_data = interpolation_candidates[:validation_data_num]
            
            # 十分な候補があれば終了
            if max_interpolation_candidates >= validation_data_num * 2:
                break
    
    if best_learning_data is None:
        raise ValueError("内挿条件を満たすデータの組み合わせが見つかりませんでした")
    
    print(f"内挿データ抽出完了: {max_interpolation_candidates}個の候補から{validation_data_num}個を選択")
    
    return best_learning_data, best_target_data


def extract_extrapolation_data(df: pd.DataFrame, learning_data_num: int, validation_data_num: int,
                              random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    外挿データ抽出
    target_dataの少なくとも1つの特徴量がlearning_dataの範囲外となるように抽出
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        learning_data_num (int): 学習データ数
        validation_data_num (int): 検証データ数
        random_seed (int): ランダムシード
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (learning_data, target_data)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if len(df) < learning_data_num + validation_data_num:
        raise ValueError(f"データが不足しています。必要: {learning_data_num + validation_data_num}, 利用可能: {len(df)}")
    
    # 内挿・外挿の対象となる特徴量を取得（分散のない列は自動除外）
    feature_columns = get_feature_columns(df)
    
    # 戦略: 複数回試行して最適なlearning_dataを見つける
    best_learning_data = None
    best_target_data = None
    max_extrapolation_candidates = 0
    
    for attempt in range(100):  # 最大100回試行
        # learning_dataをランダム選択
        current_seed = random_seed + attempt
        shuffled_df = df.sample(frac=1, random_state=current_seed).reset_index(drop=True)
        potential_learning_data = shuffled_df[:learning_data_num]
        
        # learning_dataの各特徴量の範囲を計算
        feature_ranges = {}
        for col in feature_columns:
            feature_ranges[col] = {
                'min': potential_learning_data[col].min(),
                'max': potential_learning_data[col].max()
            }
        
        # 残りのデータから外挿条件を満たすデータを探す
        remaining_data = shuffled_df[learning_data_num:]
        extrapolation_candidates = []
        
        for idx, row in remaining_data.iterrows():
            is_extrapolation = False
            for col in feature_columns:
                min_val = feature_ranges[col]['min']
                max_val = feature_ranges[col]['max']
                if row[col] < min_val or row[col] > max_val:
                    is_extrapolation = True
                    break
            
            if is_extrapolation:
                extrapolation_candidates.append(idx)
        
        extrapolation_df = remaining_data.loc[extrapolation_candidates]
        
        # 外挿データが不足している場合は、残りのデータで補完
        if len(extrapolation_df) < validation_data_num:
            non_extrapolation_data = remaining_data.drop(extrapolation_candidates)
            needed = validation_data_num - len(extrapolation_df)
            additional_data = non_extrapolation_data[:needed]
            target_candidates = pd.concat([extrapolation_df, additional_data])
        else:
            target_candidates = extrapolation_df[:validation_data_num]
        
        if len(target_candidates) >= validation_data_num:
            if len(extrapolation_df) > max_extrapolation_candidates:
                max_extrapolation_candidates = len(extrapolation_df)
                best_learning_data = potential_learning_data
                best_target_data = target_candidates[:validation_data_num]
            
            # 十分な外挿候補があれば終了
            if max_extrapolation_candidates >= validation_data_num:
                break
    
    if best_learning_data is None:
        raise ValueError("外挿条件を満たすデータの組み合わせが見つかりませんでした")
    
    print(f"外挿データ抽出完了: {max_extrapolation_candidates}個の外挿候補から構成")
    
    return best_learning_data, best_target_data


def extract_data(config: Dict) -> None:
    """
    設定に基づいてデータを抽出する
    
    Args:
        config (Dict): 設定辞書
    """
    # 設定値の取得
    input_filename = config.get('input_filename')
    output_filename = config.get('output_filename')
    input_path = config.get('input_path')
    output_path = config.get('output_path')
    random_seed = config.get('random_seed')
    extraction_pattern = config.get('extraction_pattern')
    learning_data_num = config.get('learning_data_num')
    validation_data_num = config.get('validation_data_num')
    
    # 必要なパラメータの検証
    required_params = {
        'input_filename': input_filename,
        'input_path': input_path,
        'output_path': output_path,
        'random_seed': random_seed,
        'extraction_pattern': extraction_pattern,
        'learning_data_num': learning_data_num,
        'validation_data_num': validation_data_num
    }
    
    for param_name, param_value in required_params.items():
        if param_value is None:
            raise ValueError(f"{param_name} is required")
    
    # ファイルパスの構築
    input_file_path = os.path.join(input_path, input_filename)
    
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # データの読み込み
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    df = pd.read_csv(input_file_path)
    print(f"Input data loaded: {len(df)} rows")
    
    # 抽出パターンに応じて処理を分岐
    if extraction_pattern == 1:
        print("ランダム抽出を実行中...")
        learning_data, target_data = extract_random_data(df, learning_data_num, validation_data_num, random_seed)
    elif extraction_pattern == 2:
        print("内挿データ抽出を実行中...")
        learning_data, target_data = extract_interpolation_data(df, learning_data_num, validation_data_num, random_seed)
    elif extraction_pattern == 3:
        print("外挿データ抽出を実行中...")
        learning_data, target_data = extract_extrapolation_data(df, learning_data_num, validation_data_num, random_seed)
    else:
        raise ValueError(f"Invalid extraction_pattern: {extraction_pattern}. Must be 1, 2, or 3.")
    
    # 特徴量の範囲を表示
    feature_columns = get_feature_columns(df)
    print("\n=== Learning Data 特徴量範囲 ===")
    for col in feature_columns:
        print(f"{col}: {learning_data[col].min():.1f} ~ {learning_data[col].max():.1f}")
    
    print("\n=== Target Data 特徴量範囲 ===")
    for col in feature_columns:
        print(f"{col}: {target_data[col].min():.1f} ~ {target_data[col].max():.1f}")
    
    # 出力ファイルの保存
    learning_output_path = os.path.join(output_path, 'learning_output.csv')
    target_output_path = os.path.join(output_path, 'test_output.csv')
    
    learning_data.to_csv(learning_output_path, index=False)
    target_data.to_csv(target_output_path, index=False)
    
    print(f"\nExtraction completed:")
    print(f"Learning data saved: {learning_output_path} ({len(learning_data)} rows)")
    print(f"Target data saved: {target_output_path} ({len(target_data)} rows)")


def run_extraction_from_json(json_path: str) -> None:
    """
    JSONファイルを読み込んでデータ抽出を実行
    
    Args:
        json_path (str): JSONファイルのパス
    """
    config = load_config_from_json(json_path)
    extract_data(config)


def run_extraction_from_yaml(yaml_path: str) -> None:
    """
    YAMLファイルを読み込んでデータ抽出を実行
    
    Args:
        yaml_path (str): YAMLファイルのパス
    """
    config = load_config_from_yaml(yaml_path)
    extract_data(config)


def run_extraction_from_config(config_path: str) -> None:
    """
    設定ファイルを読み込んでデータ抽出を実行（JSON/YAML自動判別）
    
    Args:
        config_path (str): 設定ファイルのパス
    """
    config = load_config(config_path)
    extract_data(config)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        run_extraction_from_config(config_path)
    else:
        print("Usage: python data_extractor.py <config_file>")
        print("Supported formats: .json, .yml, .yaml")