#!/usr/bin/env python3
"""
Add reward file generation to all test.sh files in Harbor-style task directories.

This script modifies test.sh files to write a reward (1 for pass, 0 for fail)
to /logs/verifier/reward.txt after running pytest.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


# Harbor-style test.sh template with reward file generation
TEST_SH_TEMPLATE = '''#!/bin/bash

apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
# Run pytest tests
cd /home/user
uvx \
  --python 3.12 \
  --with pytest==8.4.1 \
  pytest /tests/test_final_state.py -v
# Check exit code and write reward
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
'''


def update_test_sh(test_sh_path: Path, dry_run: bool = False) -> bool:
    """
    Update a test.sh file to include reward file generation.
    
    Returns True if the file was updated, False otherwise.
    """
    if not test_sh_path.exists():
        print(f"  ⚠️ File not found: {test_sh_path}")
        return False
    
    if dry_run:
        print(f"  Would update: {test_sh_path}")
        return True
    
    # Write the new content
    test_sh_path.write_text(TEST_SH_TEMPLATE, encoding="utf-8")
    
    # Make executable
    os.chmod(test_sh_path, 0o755)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add reward file generation to all test.sh files"
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("tasks"),
        help="Directory containing task directories (default: tasks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()
    
    task_base = Path(args.task_dir)
    
    if not task_base.exists():
        print(f"Error: Task directory not found: {task_base}")
        return
    
    # Find all task directories (directories containing "task" in name)
    task_dirs = [
        task_base / d for d in os.listdir(task_base)
        if (task_base / d).is_dir() and "task" in d.lower()
    ]
    
    print(f"Found {len(task_dirs)} task directories in {task_base}")
    
    updated_count = 0
    skipped_count = 0
    
    for task_dir in sorted(task_dirs):
        test_sh_path = task_dir / "tests" / "test.sh"
        
        if not test_sh_path.exists():
            print(f"⏭️ Skipping {task_dir.name} - no tests/test.sh found")
            skipped_count += 1
            continue
        
        if update_test_sh(test_sh_path, dry_run=args.dry_run):
            print(f"✅ Updated {task_dir.name}/tests/test.sh")
            updated_count += 1
        else:
            skipped_count += 1
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")


if __name__ == "__main__":
    main()

