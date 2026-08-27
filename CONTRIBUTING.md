# Contributing to hearthagent-pro

We welcome contributions! Here's how to get started.

## Development Setup

```bash
./scripts/setup.sh
ollama pull llama3.2:1b
ollama pull qwen2.5-coder:7b
```

## Code Style

- Python 3.10+
- Format: `black` or `ruff format`
- Lint: `ruff check`
- Type hints required

## Testing

```bash
pytest tests/
```

## Branch Naming

- `feature/` — new features
- `fix/` — bug fixes
- `docs/` — documentation
- `refactor/` — code cleanup

## Commit Messages

Format: `<type>: <description>`

```
feature: add model router classification
fix: correct cold storage archival logic
docs: update README with deployment examples
```

## Pull Requests

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit
4. Push and open a PR
5. Link any related issues

## Code Review

PRs require approval before merging. We look for:
- Clear purpose and linked issues
- Tests for new functionality
- Updated docs if needed
- No breaking changes to config.yaml schema

## Questions?

Open an issue or discussion in the repo.
