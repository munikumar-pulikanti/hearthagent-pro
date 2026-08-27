#!/bin/bash
# Append agent findings to session log

SESSION_LOG="${HOME}/.ai-memory-vault/session.log"
mkdir -p "$(dirname "$SESSION_LOG")"

echo "[$(date -Iseconds)] $*" >> "$SESSION_LOG"
