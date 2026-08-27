#!/usr/bin/env python3
"""Archive/restore memory tiers: hot → warm → cold."""

import argparse
import yaml
from pathlib import Path


def archive(days: int, dry_run: bool = False):
    """Archive old memories to cold storage."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("config.yaml not found")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    mode = config.get("mode", "local")
    cold_config = config["cold_storage"][mode]

    if dry_run:
        print(f"[DRY RUN] Would archive memories older than {days} days")
        print(f"Backend: {cold_config['backend']}")
    else:
        print(f"Archiving memories older than {days} days...")
        # TODO: Implement archival logic


def restore(row_id: str):
    """Restore memory from cold storage."""
    print(f"Restoring row {row_id} from cold storage...")
    # TODO: Implement restore logic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold storage management")
    subparsers = parser.add_subparsers(dest="command")

    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--days", type=int, default=90)
    archive_parser.add_argument("--dry-run", action="store_true")

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("row_id", type=str)

    args = parser.parse_args()

    if args.command == "archive":
        archive(args.days, args.dry_run)
    elif args.command == "restore":
        restore(args.row_id)
    else:
        parser.print_help()
