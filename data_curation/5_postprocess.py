"""Step 5: Post-processing - gather results, merge CSVs, and split train/val.

This script consolidates outputs from all previous pipeline steps:
1. gather_preprocessing: Remove errored video IDs from metadata
2. gather_filtering: Remove errored video IDs from filtering stage
3. gather_recaptioning: Merge recaptioning statistics and CSVs
4. merge_and_split: Merge all CSVs, add root_dir column, split into train/val

Usage:
    python 5_postprocess.py --input_dir /path/to/pipeline_output \
        --output_dir /path/to/final_csv --val_size 1000
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process pipeline outputs into final train/val CSVs")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Root directory containing pipeline outputs (preprocessing/, filtering/, recaptioning/ subdirs)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory for final CSV outputs")
    parser.add_argument('--val_size', type=int, default=1000,
                        help="Number of samples for validation split")
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help="Path to original metadata CSV (if not in input_dir)")
    parser.add_argument('--recap_root_dir', type=str, default=None,
                        help="Root dir value to add to final CSV for recaptioning outputs")
    return parser.parse_args()


def gather_error_videoids(error_dir):
    """Collect all error video IDs from .txt files in a directory."""
    error_videoids = set()
    if not os.path.isdir(error_dir):
        print(f"Warning: error directory not found: {error_dir}")
        return error_videoids
    for f in os.listdir(error_dir):
        if f.endswith(".txt"):
            fpath = os.path.join(error_dir, f)
            with open(fpath, 'r') as fh:
                error_videoids.update(line.strip() for line in fh if line.strip())
    return error_videoids


def gather_preprocessing(input_dir, metadata_csv, output_dir):
    """Remove errored video IDs from preprocessing and save cleaned metadata.

    Args:
        input_dir: Pipeline output root (expects input_dir/preprocessing/error/)
        metadata_csv: Path to original metadata CSV
        output_dir: Where to save cleaned CSV

    Returns:
        Path to cleaned metadata CSV
    """
    print("=== Gather Preprocessing ===")
    error_dir = os.path.join(input_dir, 'preprocessing', 'error')
    error_videoids = gather_error_videoids(error_dir)

    df = pd.read_csv(metadata_csv)
    print(f"Original metadata rows: {len(df)}")

    filtered_df = df[~df['videoid'].isin(error_videoids)]
    print(f"After removing preprocessing errors: {len(filtered_df)}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metadata_post_preprocessing.csv')
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path


def gather_filtering(input_dir, metadata_csv, output_dir):
    """Remove errored video IDs from filtering stage.

    Args:
        input_dir: Pipeline output root (expects input_dir/filtering/error/)
        metadata_csv: Path to metadata CSV (from gather_preprocessing)
        output_dir: Where to save cleaned CSV

    Returns:
        Path to cleaned metadata CSV
    """
    print("=== Gather Filtering ===")
    error_dir = os.path.join(input_dir, 'filtering', 'error')
    error_videoids = gather_error_videoids(error_dir)

    df = pd.read_csv(metadata_csv)
    print(f"Input metadata rows: {len(df)}")

    filtered_df = df[~df['videoid'].isin(error_videoids)]
    print(f"After removing filtering errors: {len(filtered_df)}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metadata_post_filtering.csv')
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return output_path


def gather_recaptioning(input_dir, output_dir):
    """Merge recaptioning statistics and gather recap CSVs.

    Args:
        input_dir: Pipeline output root (expects input_dir/recaptioning/statistics/ and recap_csv/)
        output_dir: Where to save merged outputs

    Returns:
        Path to merged recap metadata CSV, or None if no CSVs found
    """
    print("=== Gather Recaptioning ===")
    recap_dir = os.path.join(input_dir, 'recaptioning')
    stats_dir = os.path.join(recap_dir, 'statistics')
    csv_dir = os.path.join(recap_dir, 'recap_csv')
    os.makedirs(output_dir, exist_ok=True)

    # Gather statistics if available
    if os.path.isdir(stats_dir):
        # Merge num_annotation JSON files
        num_ann_files = glob.glob(os.path.join(recap_dir, 'num_annotation', '*.json'))
        if num_ann_files:
            merged_num_ann = {}
            for fp in num_ann_files:
                with open(fp, 'r') as f:
                    merged_num_ann.update(json.load(f))
            with open(os.path.join(output_dir, 'num_annotations_merged.json'), 'w') as f:
                json.dump(merged_num_ann, f, indent=4)
            print(f"Merged {len(num_ann_files)} num_annotation files -> {len(merged_num_ann)} entries")

    # Gather error files
    error_dir = os.path.join(recap_dir, 'error')
    error_videoids = gather_error_videoids(error_dir)
    print(f"Recaptioning errors: {len(error_videoids)} video IDs")

    # Merge recap CSVs if available
    if os.path.isdir(csv_dir):
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        if csv_files:
            dfs = [pd.read_csv(f) for f in csv_files]
            merged_df = pd.concat(dfs).drop_duplicates()
            output_path = os.path.join(output_dir, 'recap_merged.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Merged {len(csv_files)} recap CSVs -> {len(merged_df)} rows")
            return output_path

    return None


def merge_and_split(input_csvs, output_dir, val_size=1000, root_dir=None):
    """Merge CSV files, optionally add root_dir column, and split train/val.

    Args:
        input_csvs: List of CSV file paths to merge
        output_dir: Where to save final train/val CSVs
        val_size: Number of validation samples
        root_dir: If provided, add a 'root_dir' column with this value

    Returns:
        Tuple of (train_path, val_path)
    """
    print("=== Merge and Split ===")
    os.makedirs(output_dir, exist_ok=True)

    dfs = []
    for csv_path in input_csvs:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  Loaded {csv_path}: {len(df)} rows")
        else:
            print(f"  Warning: {csv_path} not found, skipping")

    if not dfs:
        print("No CSVs to merge!")
        return None, None

    merged_df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    print(f"Merged total: {len(merged_df)} rows")

    if root_dir:
        merged_df['root_dir'] = root_dir

    # Shuffle and split
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    actual_val_size = min(val_size, len(merged_df) // 2)
    df_val = merged_df.iloc[:actual_val_size]
    df_train = merged_df.iloc[actual_val_size:]

    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    merged_path = os.path.join(output_dir, 'all.csv')

    merged_df.to_csv(merged_path, index=False)
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"Train: {len(df_train)} rows -> {train_path}")
    print(f"Val: {len(df_val)} rows -> {val_path}")
    print(f"All: {len(merged_df)} rows -> {merged_path}")

    return train_path, val_path


if __name__ == '__main__':
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine metadata CSV
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        metadata_csv = os.path.join(input_dir, 'metadata.csv')
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(
                f"No metadata CSV found at {metadata_csv}. "
                "Please provide --metadata_csv explicitly."
            )

    # Step 1: Gather preprocessing
    post_pre_csv = gather_preprocessing(input_dir, metadata_csv, output_dir)

    # Step 2: Gather filtering
    post_filter_csv = gather_filtering(input_dir, post_pre_csv, output_dir)

    # Step 3: Gather recaptioning
    recap_csv = gather_recaptioning(input_dir, output_dir)

    # Step 4: Merge and split
    csvs_to_merge = [post_filter_csv]
    if recap_csv:
        csvs_to_merge.append(recap_csv)

    merge_and_split(
        csvs_to_merge,
        output_dir,
        val_size=args.val_size,
        root_dir=args.recap_root_dir,
    )

    print("\nPost-processing complete!")
