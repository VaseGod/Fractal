# Fractal — Self-Improving Agentic Infrastructure

A secure, self-improving multi-agent system powered by local inference (Intel Arc Pro B70), LangChain orchestration, and metacognitive feedback loops.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Bifrost API Gateway                │
│              (mTLS, Rate Limiting)                  │
├──────────┬──────────┬──────────┬───────────────────┤
│  Task    │  Meta    │  Memory  │   Benchmark       │
│  Agent   │  Agent   │ (MemCollab)│   Runner        │
│ (LangChain) │ (Evaluator) │ (ChromaDB) │ (ARC-AGI-3) │
├──────────┴──────────┼──────────┴───────────────────┤
│     HITL Gate       │    Trigger.dev Jobs          │
│   (LangSmith)       │  (Browserbase Automation)    │
├─────────────────────┴──────────────────────────────┤
│              vLLM Inference Engine                  │
│         Intel Arc Pro B70 (XPU + AWQ)              │
├────────────────────────────────────────────────────┤
│         Docker (Hardened Linux Base)               │
└────────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement | Version |
|---|---|
| Intel Arc Pro B70 GPU | B-series (Xe2 / Battlemage) |
| Intel oneAPI Base Toolkit | 2024.1+ |
| Docker & Docker Compose | 24.0+ / 2.24+ |
| Python | 3.11 |
| Node.js | 20 LTS |
| Intel GPU drivers | `intel-level-zero-gpu` |

## Quick Start

### 1. Clone and Configure

```bash
cd Fractal
cp .env.example .env
# Edit .env with your API keys and model path
```

### 2. Prepare Model Weights

Place your quantized model in the model weights directory:

```bash
mkdir -p /opt/fractal/models
# Copy your AWQ/GPTQ quantized model to /opt/fractal/models/default/
```

### 3. Build and Launch

```bash
# Build all containers
docker compose build

# Start the full stack
docker compose up -d

# Check service health
docker compose ps
docker compose logs -f inference
```

### 4. Verify Deployment

```bash
# Check vLLM inference
curl http://localhost:8000/health

# Check orchestrator
curl http://localhost:8080/health

# Check ChromaDB
curl http://localhost:8100/api/v2/heartbeat
```

## Services

| Service | Port | Description |
|---|---|---|
| `inference` | 8000 | vLLM OpenAI-compatible API (Intel XPU) |
| `orchestrator` | 8080 | Task Agent + Meta Agent (Bifrost-secured) |
| `chromadb` | 8100 | Vector database for MemCollab memory |
| `jobs` | — | Trigger.dev background job worker |

## Local Development (Without Docker)

### Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Node.js Setup

```bash
npm install
cd jobs && npm install
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Start vLLM Manually

```bash
# Ensure Intel oneAPI environment is sourced
source /opt/intel/oneapi/setvars.sh

# Launch inference
bash scripts/vllm_init.sh
```

## Security Features

- **Hardened Docker image**: Non-root user, removed setuid binaries, tini PID 1
- **Network egress filtering**: Whitelist-only outbound traffic (iptables)
- **Sandboxed execution**: Seccomp profiles + cgroups resource limits
- **Bifrost gateway**: mTLS, rate limiting, request validation, security headers
- **HITL gates**: LangSmith-integrated human approval for all destructive actions
- **Strict dependencies**: All Python/Node.js versions are pinned (no floating ranges)

## Project Structure

```
Fractal/
├── Dockerfile / docker-compose.yml     # Hardened container setup
├── requirements.txt / package.json     # Pinned dependencies
├── scripts/                            # Startup & security scripts
│   ├── vllm_init.sh                    # Intel XPU inference launcher
│   ├── network_egress_filter.sh        # iptables egress rules
│   └── sandbox_init.sh                 # Sandbox environment
├── config/                             # Service configurations
│   ├── bifrost.yaml                    # API gateway config
│   ├── vllm_config.json                # Inference engine params
│   └── langsmith.yaml                  # Tracing & HITL config
├── src/
│   ├── agents/                         # Agent orchestrators
│   │   ├── task_agent.py               # Primary LangChain agent
│   │   ├── meta_agent.py               # Metacognitive evaluator
│   │   ├── tools/                      # LangChain tools (Pydantic I/O)
│   │   └── prompts/                    # System prompts
│   ├── memory/                         # Vector memory
│   │   ├── vector_store.py             # ChromaDB operations
│   │   └── memcollab.py                # MemCollab architecture
│   ├── middleware/                      # Security middleware
│   │   ├── bifrost_router.py           # API gateway
│   │   └── hitl_gate.py                # Human approval system
│   ├── evaluation/                     # Benchmarking
│   │   ├── benchmark.py                # ARC-AGI-3 runner
│   │   ├── scoring.py                  # Efficiency scoring
│   │   └── feedback_loop.py            # Benchmark → Meta Agent
│   └── web/                            # Browser automation
│       └── browser_agent.py            # Browserbase client
├── jobs/                               # Trigger.dev TypeScript jobs
│   └── src/jobs/                       # Browser & eval background jobs
└── tests/                              # Test suite
```

## Feedback Loop

```
Benchmark Run → Efficiency Score → Compare Baseline
     ↓                                    ↓
  ARC-AGI-3 Tasks              Improved? → Archive Config
     ↓                                    ↓
  Task Agent Execution         Meta Agent Analysis
     ↓                                    ↓
  Results + Traces             Optimization Proposals
                                          ↓
                               HITL Approval → Apply Changes
```

## License

MIT — See [LICENSE](LICENSE) for details.
