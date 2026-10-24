G ?= .venv/bin/sqlagent
PY ?= .venv/bin/python
LIMIT ?= 40

.PHONY: install test lint data offline gate bench ablations redteam serve

install:
	uv venv .venv && uv pip install -p .venv -e '.[dev]'

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

data:
	$(G) data

offline: data  ## security + schema-linking evals; no API key needed
	$(G) eval security --report reports/security.md
	$(G) eval linking --report reports/linking.md

gate: offline
	$(G) eval gate --result security=reports/security.json --result linking=reports/linking.json

bench: data  ## headline ablation: direct vs agent vs agent+linking (needs ANTHROPIC_API_KEY)
	$(G) eval run --config configs/direct_full.yaml --config configs/agent_full.yaml \
	  --config configs/agent_linked.yaml --limit $(LIMIT) \
	  --out reports/agent_chinook.json --report reports/agent_chinook.md

ablations: data  ## effort and model sweeps (needs ANTHROPIC_API_KEY)
	$(G) eval run --config configs/agent_full.yaml --config configs/agent_full_low.yaml \
	  --config configs/agent_full_high.yaml --config configs/agent_sonnet.yaml --limit $(LIMIT) \
	  --out reports/agent_ablations.json --report reports/agent_ablations.md

redteam: data  ## indirect prompt injection via poisoned rows (needs ANTHROPIC_API_KEY)
	$(G) eval redteam --config configs/agent_full.yaml

serve: data
	$(G) serve --config configs/agent_full.yaml
