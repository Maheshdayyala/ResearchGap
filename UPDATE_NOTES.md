# ResearchGapAI update notes

This update is based on the uploaded project archive. Local `.env` credentials and virtual-environment/cache directories are intentionally excluded from the delivered ZIP.

## Changes included
- Simplified the landing-page hero and removed the large marketing headline/capability strip.
- Removed the frontend LLM configuration disclosure section; backend environment-based LLM configuration remains intact.
- Applied the selected deep-navy / blue-cyan theme across the application.
- Added an analysis-scoped conversational assistant UI and API. Messages are persisted in PostgreSQL and scoped to the authenticated user and analysis. The assistant requires the server's configured LLM provider; if none is configured, the API reports that clearly rather than pretending to answer.
- Updated History page wording and retained links to reopen saved reports.
- Replaced the static Evaluation page's title/intro with an Evaluation Dashboard framing.
- Added a `chat_messages` table model; `db.create_all()` creates this new table on application startup without deleting existing data.

## Important items still requiring validation / additional implementation
- Existing users/searches are already represented by `users`, `searches`, `papers`, and `gap_reports`; password hashes use Werkzeug. Existing `db.create_all()` does not migrate altered columns, so any future schema changes should use Alembic/Flask-Migrate.
- Agent-wise benchmark metrics (precision/recall/F1, evidence faithfulness, graph edge accuracy) cannot honestly be computed without a labelled benchmark or human adjudication. This update does not invent percentages or scores. The evaluation page still needs a full persisted metrics dashboard and graph visualizations based on benchmark data.
- Research-gap detection and novelty verification require further empirical tuning against the three user-provided topics and a manually reviewed paper set. The current heuristic/lexicon-based detector and prior-art checks are retained; no claim of validated gap-detection improvement is made here.
- HTTPS cannot be made trustworthy for public deployment without a domain and certificate/reverse-proxy configuration. Local `http://localhost` remains a development endpoint.
- The complete UI/chat changes need to be rebuilt and tested in the user's Docker environment with the user's own `.env` file.

## Run after replacing the project files
Use the existing project `.env` locally (do not share it). Then run `docker compose up -d --build` and check `docker compose ps`. Existing named database volumes are not removed by this update.
