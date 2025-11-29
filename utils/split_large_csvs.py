#!/usr/bin/env python3
"""
Split large CSV files into smaller chunks for GitHub upload.

This script:
1. Creates a backup of original files in data/originals/
2. Splits large CSV files into chunks (default: 50 MB each)
3. Creates split files in data/split_files/ 
4. Does NOT modify or delete any original files
5. Updates .gitignore to exclude originals

Usage:
    python utils/split_large_csvs.py --max-size 50
"""

import argparse
import glob
import logging
import os
import shutil
from datetime import datetime
from typing import List

import pandas as pd


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)


def backup_file(filepath: str, backup_dir: str) -> str:
    """Safely backup a file to the backup directory."""
    os.makedirs(backup_dir, exist_ok=True)
    filename = os.path.basename(filepath)
    backup_path = os.path.join(backup_dir, filename)
    
    if os.path.exists(backup_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        backup_path = os.path.join(backup_dir, f"{name}_backup_{timestamp}{ext}")
    
    logging.info("Backing up %s to %s", filepath, backup_path)
    shutil.copy2(filepath, backup_path)
    return backup_path


def split_csv_file(filepath: str, output_dir: str, max_size_mb: float = 50) -> List[str]:
    """Split a CSV file into smaller chunks by rows."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    logging.info("Reading %s...", filepath)
    df = pd.read_csv(filepath)
    total_rows = len(df)
    total_size_mb = get_file_size_mb(filepath)
    
    # Estimate rows per chunk
    rows_per_mb = total_rows / total_size_mb
    chunk_rows = int(rows_per_mb * max_size_mb * 0.9)  # 0.9 safety factor
    
    if chunk_rows >= total_rows:
        logging.info("File %s (%.2f MB) is already under the size limit", filename, total_size_mb)
        return []
    
    num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
    logging.info("Splitting %s (%.2f MB, %d rows) into %d chunks of ~%d rows each",
                 filename, total_size_mb, total_rows, num_chunks, chunk_rows)
    
    split_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_rows
        end_idx = min((i + 1) * chunk_rows, total_rows)
        chunk_df = df.iloc[start_idx:end_idx]
        
        split_filename = f"{name}_part{i+1:03d}{ext}"
        split_path = os.path.join(output_dir, split_filename)
        
        logging.info("Writing chunk %d/%d: %s (%d rows)", i+1, num_chunks, split_filename, len(chunk_df))
        chunk_df.to_csv(split_path, index=False)
        
        chunk_size_mb = get_file_size_mb(split_path)
        logging.info("  Created %s (%.2f MB)", split_filename, chunk_size_mb)
        split_files.append(split_path)
    
    return split_files


def update_gitignore(repo_root: str, patterns: List[str]) -> None:
    """Update .gitignore with patterns to exclude original files."""
    gitignore_path = os.path.join(repo_root, '.gitignore')
    
    # Read existing gitignore
    existing_patterns = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_patterns = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
    
    # Add new patterns
    new_patterns = [p for p in patterns if p not in existing_patterns]
    
    if new_patterns:
        logging.info("Updating .gitignore with %d new patterns", len(new_patterns))
        with open(gitignore_path, 'a') as f:
            f.write("\n# Large CSV files - originals backed up, use split versions\n")
            for pattern in new_patterns:
                f.write(f"{pattern}\n")
                logging.info("  Added to .gitignore: %s", pattern)
    else:
        logging.info("All patterns already in .gitignore")


def main():
    parser = argparse.ArgumentParser(description="Split large CSV files for GitHub")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--pattern", default="Cascadia*.csv", help="File pattern (default: Cascadia*.csv)")
    parser.add_argument("--max-size", type=float, default=50, help="Max MB per chunk (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--update-gitignore", action="store_true", default=True, 
                       help="Update .gitignore to exclude originals (default: True)")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    
    # Setup directories
    data_dir = os.path.abspath(args.data_dir)
    backup_dir = os.path.join(data_dir, "originals")
    split_dir = os.path.join(data_dir, "split_files")
    repo_root = os.path.dirname(data_dir)  # Assume data is one level below repo root
    
    # Find files to process
    pattern = os.path.join(data_dir, args.pattern)
    files = glob.glob(pattern)
    
    # Exclude intermediate files and already split files
    files = [f for f in files if '_intermediate_' not in f and '_part' not in f]
    
    if not files:
        logging.error("No files found matching pattern: %s", pattern)
        return
    
    logging.info("Found %d files matching pattern '%s'", len(files), args.pattern)
    
    # Analyze files
    files_to_split = []
    for filepath in files:
        size_mb = get_file_size_mb(filepath)
        filename = os.path.basename(filepath)
        
        if size_mb > args.max_size:
            logging.info("  %s: %.2f MB - WILL SPLIT", filename, size_mb)
            files_to_split.append(filepath)
        else:
            logging.info("  %s: %.2f MB - OK (under limit)", filename, size_mb)
    
    if not files_to_split:
        logging.info("No files need splitting (all under %.2f MB)", args.max_size)
        return
    
    if args.dry_run:
        logging.info("\nDRY RUN - no files were modified")
        logging.info("Would split %d files:", len(files_to_split))
        for filepath in files_to_split:
            logging.info("  - %s (%.2f MB)", os.path.basename(filepath), get_file_size_mb(filepath))
        
        if args.update_gitignore:
            logging.info("\nWould add to .gitignore:")
            logging.info("  data/originals/")
            for filepath in files_to_split:
                filename = os.path.basename(filepath)
                logging.info("  data/%s", filename)
        return
    
    # Process files
    split_info = {}
    gitignore_patterns = ["data/originals/"]
    
    for filepath in files_to_split:
        filename = os.path.basename(filepath)
        logging.info("\n" + "="*60)
        logging.info("Processing: %s", filename)
        logging.info("="*60)
        
        # Backup original
        backup_path = backup_file(filepath, backup_dir)
        
        # Split file
        split_files = split_csv_file(filepath, split_dir, max_size_mb=args.max_size)
        
        if split_files:
            split_info[filename] = split_files
            logging.info("Successfully split %s into %d parts", filename, len(split_files))
            
            # Add pattern to gitignore
            gitignore_patterns.append(f"data/{filename}")
    
    # Update .gitignore
    if split_info and args.update_gitignore:
        update_gitignore(repo_root, gitignore_patterns)
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info("Backups saved to: %s", backup_dir)
    logging.info("Split files saved to: %s", split_dir)
    logging.info("Original files: NOT modified or deleted")
    
    if args.update_gitignore:
        logging.info("\n.gitignore updated to exclude:")
        for pattern in gitignore_patterns:
            logging.info("  - %s", pattern)
    
    logging.info("\nYou can now:")
    logging.info("  1. git add data/split_files/")
    logging.info("  2. git commit -m 'Add split CSV files for GitHub'")
    logging.info("  3. git push")
    logging.info("\nTo reconstruct original files later:")
    logging.info("  Use: pd.concat([pd.read_csv(f) for f in sorted(glob.glob('data/split_files/{name}_part*.csv'))], ignore_index=True)")


if __name__ == "__main__":
    main()
