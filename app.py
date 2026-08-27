"""Flask web UI for memory browser and management."""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from pathlib import Path

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return jsonify({"status": "hearthagent-pro memory vault", "version": "0.1.0"})


@app.route("/api/memories", methods=["GET"])
def get_memories():
    """List memories with filters."""
    # TODO: Query global_brain.db for memories
    return jsonify({"memories": [], "count": 0})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Memory statistics."""
    return jsonify({"total": 0, "by_scope": {}, "by_confidence": {}})


@app.route("/api/cold-storage", methods=["GET"])
def cold_storage_status():
    """Cold storage archive and warmup stats."""
    return jsonify({"archived": 0, "warmup_events": []})


@app.route("/api/savings", methods=["GET"])
def savings_report():
    """Task savings report."""
    return jsonify({"by_type": {}, "total_savings": 0})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5555, debug=True)
