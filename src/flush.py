#!/usr/bin/env python3
"""
Quick cleanup script for evaluation artifacts.
Empties (but keeps) the following folders:
  - generated_solutions/
  - prompts/
  - prompts_correct/
  - solutions_correct/
"""

import os
import shutil

TARGET_DIRS = [
    "generated_solutions",
    "prompts",
    "prompts_correct",
    "solutions_correct",
]
SUBFOLDERS = ["openai", "gemini"]


def flush_dir(path: str):
    """Delete all contents inside a directory, but keep the directory itself."""
    if os.path.exists(path):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            try:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.unlink(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            except Exception as e:
                print(f"⚠️  Error deleting {full_path}: {e}")
    else:
        os.makedirs(path, exist_ok=True)


def recreate_structure():
    """Ensure each main dir and its subfolders exist."""
    for d in TARGET_DIRS:
        os.makedirs(d, exist_ok=True)
        for sub in SUBFOLDERS:
            os.makedirs(os.path.join(d, sub), exist_ok=True)


def main():
    for d in TARGET_DIRS:
        flush_dir(d)
    recreate_structure()
    print("✅ All prompt and solution folders emptied but preserved.")


if __name__ == "__main__":
    main()
