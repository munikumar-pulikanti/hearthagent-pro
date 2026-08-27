"""Cold storage: archive idle memories to MinIO/S3, restore on demand.
Idle proxy: rows not updated in --days days (the current schema has no
separate last-accessed timestamp, so updated_at is the closest signal)."""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import boto3
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
VAULT_DB = Path.home() / ".ai-memory-vault" / "global_brain.db"
CHROMA_PATH = Path.home() / ".ai-memory-vault" / "chroma"


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["cold_storage"][cfg.get("mode", "local")]


def get_client(cfg):
    return boto3.client(
        "s3",
        endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key", ""),
        aws_secret_access_key=cfg.get("secret_key", ""),
    )


def ensure_bucket(client, bucket):
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


def ensure_columns():
    conn = sqlite3.connect(VAULT_DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    if "archived" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN archived INTEGER DEFAULT 0")
    if "cold_key" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN cold_key TEXT")
    conn.commit()
    conn.close()


def find_idle(days):
    ensure_columns()
    cutoff = int(time.time()) - days * 86400
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM memories WHERE archived = 0 AND COALESCE(updated_at, created_at) < ?",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def archive(days, dry_run):
    candidates = find_idle(days)
    if not candidates:
        print(f"No memories idle for {days}+ days.")
        return

    print(f"{'Would archive' if dry_run else 'Archiving'} {len(candidates)} memories:")
    for row in candidates:
        print(f"  [{row['id']}] {row['scope']}/{row['type']}: {row['content'][:60]}")

    if dry_run:
        print("\nDry run -- nothing changed. Re-run without --dry-run to actually archive.")
        return

    cfg = load_config()
    client = get_client(cfg)
    ensure_bucket(client, cfg["bucket"])

    conn = sqlite3.connect(VAULT_DB)
    for row in candidates:
        key = f"memories/{row['id']}.json"
        client.put_object(
            Bucket=cfg["bucket"], Key=key,
            Body=json.dumps(row).encode("utf-8"),
            ContentType="application/json",
        )
        conn.execute("UPDATE memories SET archived = 1, cold_key = ? WHERE id = ?", (key, row["id"]))
    conn.commit()
    conn.close()

    try:
        import chromadb
        chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = chroma.get_or_create_collection("memories")
        collection.delete(ids=[str(r["id"]) for r in candidates])
    except Exception as e:
        print(f"Warning: could not remove archived rows from ChromaDB: {e}")

    print(f"\nArchived {len(candidates)} memories to bucket '{cfg['bucket']}'.")


def restore(row_id):
    conn = sqlite3.connect(VAULT_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (row_id,)).fetchone()
    if not row:
        print(f"No memory with id {row_id}.")
        conn.close()
        return
    if not row["archived"]:
        print(f"Memory {row_id} is not archived.")
        conn.close()
        return

    cfg = load_config()
    client = get_client(cfg)
    obj = client.get_object(Bucket=cfg["bucket"], Key=row["cold_key"])
    data = json.loads(obj["Body"].read())

    conn.execute("UPDATE memories SET archived = 0 WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = chroma.get_or_create_collection("memories")
        embedding = embedder.encode([data["content"]]).tolist()
        collection.upsert(
            ids=[str(row_id)], documents=[data["content"]],
            metadatas=[{"scope": data["scope"], "type": data["type"]}],
            embeddings=embedding,
        )
    except Exception as e:
        print(f"Warning: could not re-index into ChromaDB: {e}")

    print(f"Restored memory {row_id}: {data['content'][:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_archive = sub.add_parser("archive")
    p_archive.add_argument("--days", type=int, default=90)
    p_archive.add_argument("--dry-run", action="store_true")

    p_restore = sub.add_parser("restore")
    p_restore.add_argument("row_id", type=int)

    args = parser.parse_args()
    if args.command == "archive":
        archive(args.days, args.dry_run)
    elif args.command == "restore":
        restore(args.row_id)
    else:
        parser.print_help()
