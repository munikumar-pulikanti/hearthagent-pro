"""Backend abstraction for embeddings, cold storage, and team sync."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import boto3


class BackendConfig:
    """Load and manage backend configuration."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.mode = self.config.get("mode", "local")

    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings backend config for current mode."""
        return self.config["embeddings"][self.mode]

    def get_cold_storage_config(self) -> Dict[str, Any]:
        """Get cold storage config for current mode."""
        return self.config["cold_storage"][self.mode]

    def get_team_sync_config(self) -> Dict[str, Any]:
        """Get team sync (warm tier) config for current mode."""
        return self.config["team_sync"][self.mode]

    def get_router_config(self) -> Dict[str, Any]:
        """Get model router config for current mode."""
        return self.config["router"][self.mode]


class S3Client:
    """S3-compatible client for cold storage (local MinIO, AWS S3, R2, B2, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config["backend"]
        self.endpoint = config.get("endpoint")
        self.bucket = config.get("bucket")
        
        # Initialize boto3 client
        kwargs = {"service_name": "s3"}
        if self.endpoint:
            kwargs["endpoint_url"] = self.endpoint
        if config.get("access_key"):
            kwargs["aws_access_key_id"] = config["access_key"]
            kwargs["aws_secret_access_key"] = config["secret_key"]
        
        self.client = boto3.client(**kwargs)

    def upload(self, key: str, data: bytes) -> bool:
        """Upload data to cold storage."""
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download(self, key: str) -> Optional[bytes]:
        """Download data from cold storage."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    def list_objects(self, prefix: str = "") -> list:
        """List objects in cold storage."""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix
            )
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            print(f"List failed: {e}")
            return []
