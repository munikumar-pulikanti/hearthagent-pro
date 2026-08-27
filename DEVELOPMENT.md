# Development Roadmap

## Phase 1: Core (Current)
- [x] Repository setup
- [x] Config abstraction layer
- [x] Model router skeleton
- [x] Cold storage skeleton
- [x] Savings tracker with logging
- [x] Flask web UI skeleton
- [ ] **Wire Ollama client integration**
- [ ] **Implement classifier + routing logic**
- [ ] **Test local model routing**

## Phase 2: Memory Tiers
- [ ] ChromaDB integration
- [ ] Semantic search implementation
- [ ] Auto-warmup logic
- [ ] Cold storage archive/restore
- [ ] Warm tier (Turso/MongoDB)
- [ ] Memory CRUD operations

## Phase 3: Evals & Quality
- [ ] LLM-as-judge eval loop
- [ ] Quality scoring (1-5)
- [ ] Precision/recall alerts
- [ ] Savings tracking visualization
- [ ] Report generation

## Phase 4: Production
- [ ] Cloud backend adapters (AWS, GCP, Azure)
- [ ] LiteLLM integration (multi-cloud LLMs)
- [ ] Web UI frontend (React/Vue)
- [ ] Deployment automation (CloudFormation, Terraform)
- [ ] Monitoring & observability

## Phase 5: Advanced
- [ ] Multi-agent coordination
- [ ] Distributed memory sync
- [ ] RAG fine-tuning
- [ ] Custom classifier training
- [ ] Cost analytics

---

## How to Contribute

Pick a task from any phase and open an issue + PR. See [CONTRIBUTING.md](./CONTRIBUTING.md).
