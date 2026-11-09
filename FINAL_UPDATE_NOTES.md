# ResearchGapAI – Final update notes

## UI
- Restored the original light interface from the supplied project archive.
- Removed the navy/blue/cyan theme overrides.
- Removed the LLM configuration panel from the user-facing page; server-side `.env` configuration remains supported.
- Removed the standalone Evaluation navigation item and page.
- Added the analysis-scoped Research Assistant tab to the existing analysis UI.

## Saved analysis and history
- History opens saved reports by report ID, including reports whose pipeline status is not `done` when a report record exists.
- Report access, chat access, and paper access continue to enforce the signed-in user's ownership.
- Chat turns are persisted in PostgreSQL and associated with both the user and analysis.
- The analysis report includes a per-agent execution summary with actual output counts/statuses. These are execution records, not fabricated benchmark scores.

## Research-gap quality safeguards
- Topic matching now requires multiple topical anchors rather than a single shared word.
- The AAI acronym collision with animal-assisted intervention papers is explicitly guarded when the query does not ask for animal-assisted interventions.
- If too few topic-matched prior-art records are available, novelty is labelled `INSUFFICIENT_EVIDENCE`; it is not accepted as novel.
- Experiment planning no longer pads a plan with loosely related Hugging Face datasets or generic accuracy metrics when task fit is unknown.
- Planning a candidate no longer automatically marks that candidate as a verified research gap.
- Partially explored candidates can still receive a clearly labelled provisional experiment plan through the workflow fallback.

## Data and deployment
- Existing SQLAlchemy models are retained; `ChatMessage` is created by the existing `db.create_all()` startup behavior.
- Existing PostgreSQL/Neo4j Docker Compose services and healthchecks are retained.
- Keep your existing `.env` file. It is intentionally excluded from the ZIP.
- Do not use `docker compose down -v`; it deletes named database volumes.

## Validation
- Python syntax and JavaScript syntax are checked before packaging.
- External scholarly APIs and configured LLM providers must be tested in the user's own environment.
- A novelty verdict is a corpus-scoped prior-art assessment, not proof of global novelty. Human review of cited papers remains required.
