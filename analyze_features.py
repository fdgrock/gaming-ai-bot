#!/usr/bin/env python3
"""
Deep analysis of feature files and training code
Maps feature files to model requirements
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

def analyze_features():
    """Analyze all feature files in data/features"""
    features_dir = Path('data/features')
    
    analysis = {
        'current_features': defaultdict(lambda: defaultdict(list)),
        'backup_features': defaultdict(lambda: defaultdict(list)),
    }
    
    # Analyze current features
    for model_type_dir in features_dir.iterdir():
        if not model_type_dir.is_dir() or model_type_dir.name == 'Backup Features':
            continue
            
        model_type = model_type_dir.name
        
        for game_dir in model_type_dir.iterdir():
            if not game_dir.is_dir():
                continue
                
            game = game_dir.name
            
            # CSV files
            for csv_file in game_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    feature_count = len(numeric_cols)
                    analysis['current_features'][model_type][game].append({
                        'file': csv_file.name,
                        'type': 'csv',
                        'total_cols': len(df.columns),
                        'numeric_cols': feature_count,
                        'size_mb': csv_file.stat().st_size / (1024*1024)
                    })
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
            
            # NPZ files
            for npz_file in game_dir.glob('*.npz'):
                try:
                    data = np.load(npz_file)
                    key = list(data.keys())[0] if 'features' not in data and 'X' not in data else ('features' if 'features' in data else 'X')
                    arr = data[key]
                    feature_count = arr.shape[1] if len(arr.shape) > 1 else arr.shape[0]
                    analysis['current_features'][model_type][game].append({
                        'file': npz_file.name,
                        'type': 'npz',
                        'shape': str(arr.shape),
                        'feature_count': feature_count,
                        'size_mb': npz_file.stat().st_size / (1024*1024)
                    })
                except Exception as e:
                    print(f"Error reading {npz_file}: {e}")
            
            # Meta files
            for meta_file in game_dir.glob('*.meta.json'):
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        if 'feature_count' in meta:
                            analysis['current_features'][model_type][game][-1]['metadata_feature_count'] = meta['feature_count']
                except:
                    pass
    
    # Analyze backup features
    backup_dir = features_dir / 'Backup Features'
    if backup_dir.exists():
        for model_type_dir in backup_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
                
            model_type = model_type_dir.name
            
            for game_dir in model_type_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                    
                game = game_dir.name
                
                for csv_file in game_dir.glob('*.csv'):
                    try:
                        df = pd.read_csv(csv_file, nrows=1)
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        feature_count = len(numeric_cols)
                        analysis['backup_features'][model_type][game].append({
                            'file': csv_file.name,
                            'type': 'csv',
                            'total_cols': len(df.columns),
                            'numeric_cols': feature_count
                        })
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                
                for npz_file in game_dir.glob('*.npz'):
                    try:
                        data = np.load(npz_file)
                        key = list(data.keys())[0]
                        arr = data[key]
                        feature_count = arr.shape[1] if len(arr.shape) > 1 else arr.shape[0]
                        analysis['backup_features'][model_type][game].append({
                            'file': npz_file.name,
                            'type': 'npz',
                            'shape': str(arr.shape),
                            'feature_count': feature_count
                        })
                    except Exception as e:
                        print(f"Error reading {npz_file}: {e}")
    
    return analysis


def print_analysis(analysis):
    """Print detailed analysis"""
    print("\n" + "="*80)
    print("CURRENT FEATURES ANALYSIS")
    print("="*80)
    
    for model_type in sorted(analysis['current_features'].keys()):
        print(f"\n[{model_type.upper()}]")
        for game in sorted(analysis['current_features'][model_type].keys()):
            print(f"  {game}:")
            for item in analysis['current_features'][model_type][game]:
                if item['type'] == 'csv':
                    print(f"    - {item['file']}")
                    print(f"      Total columns: {item['total_cols']}, Numeric: {item['numeric_cols']}")
                    print(f"      Size: {item['size_mb']:.2f} MB")
                else:
                    print(f"    - {item['file']}")
                    print(f"      Shape: {item['shape']}, Features: {item['feature_count']}")
                    print(f"      Size: {item['size_mb']:.2f} MB")
                if 'metadata_feature_count' in item:
                    print(f"      Metadata feature count: {item['metadata_feature_count']}")
    
    print("\n" + "="*80)
    print("BACKUP FEATURES ANALYSIS")
    print("="*80)
    
    for model_type in sorted(analysis['backup_features'].keys()):
        print(f"\n[{model_type.upper()}]")
        for game in sorted(analysis['backup_features'][model_type].keys()):
            print(f"  {game}:")
            for item in analysis['backup_features'][model_type][game]:
                if item['type'] == 'csv':
                    print(f"    - {item['file']}: {item['numeric_cols']} numeric features")
                else:
                    print(f"    - {item['file']}: {item['feature_count']} features (shape: {item['shape']})")


if __name__ == "__main__":
    print("Analyzing feature files...")
    analysis = analyze_features()
    print_analysis(analysis)
