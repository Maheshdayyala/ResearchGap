# ResearchGapAI: Autonomous Multi-Agent Framework for Research Gap Discovery

Implements the architecture in the *Design and Solution* document (Presentation → Agent Orchestration → Intelligence → Data layers).

| Design component | Code |
|---|---|
| Flask REST API + Flask-Login, CSRF, rate-limit (Table 3 endpoints) | `app.py`, `models.py`, `services.py` |
| LangGraph orchestration (state, routing, retry loops, Fig. 2) | `workflow.py` |
| Literature Search Agent (S2, arXiv, CrossRef, OpenAlex; parallel, dedupe, rank, query refinement, citation expansion) | `agents/literature_search.py` |
| Paper Understanding Agent (LLM JSON extraction, heuristic fallback) | `agents/paper_understanding.py` |
| Knowledge Graph Builder (Neo4j + NetworkX: USES, CITES, SIMILAR_TO, CONTRADICTS) | `agents/knowledge_graph.py` |
| Research Gap Detection Agent (unexplored method–dataset pairs, contradictions, sparse topics, repeated limitations) | `agents/gap_detection.py` |
| Novelty Verification Agent, `N(g)=0.40Nsem+0.25Ncov+0.20Nrec+0.15Nevi` | `agents/novelty.py` |
| Experiment Planning Agent | `agents/experiment_planning.py` |
| Report Generation Module | `agents/report.py` |
| Embeddings / contradiction classifier | `agents/embeddings.py`, `agents/contradiction.py` |
| Evaluation workflows (Table 4) | `evaluation/` |
| Docker / Postgres / Neo4j deployment | `Dockerfile`, `docker-compose.yml` |

## Run
```powershell
# Python 3.11 recommended on Windows
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Copy-Item .env.example .env  # optional: configure an LLM provider in .env
python app.py
```
Open http://127.0.0.1:5000. To enable LLM-backed reasoning, edit the local `.env` file with your own provider key/model, then restart the server. Keep `.env` private and never commit it. `docker compose up --build` is also supported for the containerized stack. Run `pytest -q` for offline tests.
Register an account at `/register`, enter a topic, and watch the pipeline run. Set `ADMIN_EMAIL` to make that account an admin (`/admin/stats`).

## REST API (all require login; JSON; send `X-CSRFToken` from the page's meta tag)
`POST /search` · `GET /papers/{id}` · `POST /understand` · `POST /gap-detect` · `POST /verify-novelty` · `POST /experiment-plan` · `GET /report/{id}[?format=md|html]`
Plus `POST /pipeline` (runs the whole LangGraph workflow in the background) and `GET /status/{search_id}`.

## Modes and honest limitations
* **No LLM key:** every agent still runs using rule-based fallbacks (regex extraction, template gap statements, polarity-based contradiction detection). Quality is clearly lower than with an LLM; use a key for real results.
* **Embeddings:** Sentence-Transformers `all-MiniLM-L6-v2` if installed; otherwise a hashed TF vector fallback (similarity values are lower, so re-tune `NOVELTY_THRESHOLD`).
* **Novelty weights and θ are initial values** (per the design); tune them with `evaluation/backtest.py`. Only abstracts/metadata are processed, never full text.

## HTTPS and production deployment

Local development can continue over `http://localhost:5000`; `COOKIE_SECURE` defaults to `false` for that workflow. For deployment, terminate TLS at a trusted reverse proxy (for example Caddy, nginx, or a managed load balancer), forward the original scheme using `X-Forwarded-Proto: https`, and set `COOKIE_SECURE=true`. The app trusts exactly one proxy hop, sets `Secure` session cookies in that mode, and emits HSTS after HTTPS requests. Do not expose the app port publicly around the TLS proxy. Set a long random `SECRET_KEY` and unique `POSTGRES_PASSWORD` in the deployment environment before starting the stack.

PostgreSQL data is stored in the named `pgdata` volume, so accounts, analyses, reports, evaluations, and assistant conversations survive container restarts. Removing that volume removes this data.

## Evaluation interpretation

Per-analysis and historical dashboards report persisted agent execution counts (papers retrieved and structured, graph elements, candidate checks, and plan steps). They do not invent precision, recall, F1, or evidence-accuracy percentages. Those benchmark metrics require a labelled reference set; the evaluation modules under `evaluation/` provide the benchmark workflows to produce them.
* **Neo4j** is optional: the graph is always built in NetworkX; if `NEO4J_URI` is set it is also written to Neo4j and the unexplored-pair query runs in Cypher.
* **Evaluation:** `llm_judge.py` implements the rubric judges directly; Ragas/DeepEval are not wired in, but can consume the same inputs. Sample sizes in the design are small, so report results as indicative.
* Outputs are recommendations that need human validation.
* Not verified here: live calls to the four scholarly APIs, LLM calls, Neo4j and Postgres (the offline tests cover the pipeline logic, the REST auth flow and the DB models on SQLite).

## v2 changes (what and why)
* **Concrete gaps.** A gap is now a specific, checkable combination (e.g. *federated learning × fairness*): two concepts each studied in ≥2 primary studies that never co-occur, a concern raised in reviews but never tested, a genuine contradiction between primary studies, or a concrete repeated limitation. Reviews are detected and excluded from "studied together" counts.
* **Novelty = prior-art search, not a threshold.** Targeted queries across the four APIs + citation network → every paper is checked against *all* core concepts → verdict `NOVEL` / `PARTIALLY_EXPLORED` / `ALREADY_ADDRESSED` + confidence + closest papers (shown with which concepts matched). With an LLM the closest papers are also judged. The design-document score N(g) is kept only as a ranking signal.
* **Grounded experiment plans.** Datasets come from `agents/lexicon.py` (curated registry with links and access terms), datasets found in the retrieved papers, and a best-effort Hugging Face search. Baselines/metrics are task- and aspect-specific. Nothing is free-generated; with an LLM it may only choose from these candidate lists.
* **LLM options.** `LLM_BASE_URL` + `LLM_API_KEY` works with any OpenAI-compatible API (Groq, Gemini, OpenRouter, a local Ollama at `http://localhost:11434/v1`).
* Abstract HTML is stripped; noisy method extraction (`covid`, `mri`, `ai-based`) is filtered.
* Delete the old `instance/researchgapai.db` (or `researchgapai.db`) after upgrading: stored reports from v1 use the old format.

## PostgreSQL + Neo4j setup (design-document architecture)

The application uses PostgreSQL for relational records (users, searches, papers,
citations and generated gap reports) and Neo4j for the research knowledge graph.
The in-process NetworkX graph remains the graph-construction workspace and fallback;
when Neo4j is configured, the graph builder mirrors each completed search graph to Neo4j.

### Start the data services with Docker Desktop

1. Copy `.env.example` to `.env` and set a strong `SECRET_KEY` and your LLM provider key.
2. From this project directory, run `docker compose up -d postgres neo4j`.
3. Activate the Python 3.11 virtual environment and install requirements:
   `python -m pip install -r requirements.txt`.
4. Ensure `.env` contains `DATABASE_URL=postgresql://rgai:rgai@localhost:5432/researchgapai`,
   `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, and the matching
   `NEO4J_PASSWORD` (default in the local compose file is `researchgap`).
5. Run `python app.py`. Flask creates the relational tables on startup.
6. Visit `http://127.0.0.1:5000/health` and confirm the database connection and Neo4j
   connection fields report `connected: true`.

To run the complete stack in containers, configure `.env` and run `docker compose up --build`.
The web container connects to PostgreSQL and Neo4j by their Compose service names.

For a lightweight code-only run, leaving `DATABASE_URL` unset retains the SQLite fallback;
Neo4j is then optional and graph reasoning can fall back to NetworkX. This fallback is for
local development, not the database configuration specified by the project design.

**Local development credentials are not production credentials.** Change the PostgreSQL,
Neo4j and Flask secrets before deploying beyond a private development machine.
