"""Central configuration, read from environment variables."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load project-local .env without overriding existing environment variables.
except ImportError:
    pass

# ======================= EDIT HERE (no .env file needed) =======================
# Fill in only what you use; leave the rest as "". Environment variables, if set, override these.
LOCAL = {
    # --- LLM: pick ONE option (leave all empty to run in rule-based mode) ---
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LLM_API_KEY": "",        # key for any OpenAI-compatible API (OpenAI, Groq, Gemini, OpenRouter)
    "LLM_BASE_URL": "",       # e.g. https://api.groq.com/openai/v1 | http://localhost:11434/v1 (Ollama, no key needed)
    "LLM_MODEL": "",          # e.g. claude-haiku-5-5 | llama-3.3-70b-versatile | gpt-4o-mini
    # --- optional services ---
    "NEO4J_URI": "", "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "",
    "DATABASE_URL": "",       # empty = local SQLite file
    "S2_API_KEY": "", "CONTACT_EMAIL": "you@example.com",
    "SECRET_KEY": "change-me-to-any-random-string", "ADMIN_EMAIL": "",
}
# ===============================================================================


def env(name, default=""):
    return os.getenv(name) or LOCAL.get(name) or default


def _db_url():
    url = env("DATABASE_URL", "sqlite:///researchgapai.db")
    return url.replace("postgres://", "postgresql://", 1)


class Settings:
    SECRET_KEY = env("SECRET_KEY", "dev-change-me")
    DATABASE_URL = _db_url()
    ADMIN_EMAIL = env("ADMIN_EMAIL", "").lower()
    COOKIE_SECURE = env("COOKIE_SECURE", "false").lower() in ("1", "true", "yes", "on")

    # LLM
    ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = env("OPENAI_API_KEY", "")
    LLM_PROVIDER = env("LLM_PROVIDER", "auto").lower()  # auto | anthropic | openai | none
    LLM_MODEL = env("LLM_MODEL", "")
    LLM_BASE_URL = env("LLM_BASE_URL", "")      # any OpenAI-compatible API: Groq, Gemini, OpenRouter, Ollama (http://localhost:11434/v1)
    LLM_API_KEY = env("LLM_API_KEY", "")

    # Graph DB
    NEO4J_URI = env("NEO4J_URI", "")
    NEO4J_USER = env("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = env("NEO4J_PASSWORD", "")

    # Scholarly APIs
    S2_API_KEY = env("S2_API_KEY", "")
    CONTACT_EMAIL = env("CONTACT_EMAIL", "researchgapai@example.com")

    # Embeddings / NLI
    EMBED_MODEL = env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    USE_SBERT = env("USE_SBERT", "auto")        # auto | true | false
    USE_NLI = env("USE_NLI", "false").lower() == "true"
    NLI_MODEL = env("NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")

    # Pipeline
    MIN_PAPERS = int(env("MIN_PAPERS", 12))
    MAX_PAPERS = int(env("MAX_PAPERS", 60))
    PER_SOURCE = int(env("PER_SOURCE", 25))
    MAX_REFINEMENTS = int(env("MAX_REFINEMENTS", 3))
    MAX_DETECT_ROUNDS = int(env("MAX_DETECT_ROUNDS", 3))
    SIM_EDGE = float(env("SIM_EDGE", 0.45))
    CONTRADICTION_MIN = float(env("CONTRADICTION_MIN", 0.6))

    # Novelty  N(g) = W_SEM*Nsem + W_COV*Ncov + W_REC*Nrec + W_EVI*Nevi   (initial weights; tune empirically)
    NOVELTY_THRESHOLD = float(env("NOVELTY_THRESHOLD", 0.55))
    W_SEM, W_COV, W_REC, W_EVI = 0.40, 0.25, 0.20, 0.15
    DIRECT_SIM = float(env("DIRECT_SIM", 0.72))
    RECENT_YEARS = int(env("RECENT_YEARS", 3))
    MAX_VERIFY = int(env("MAX_VERIFY", 6))          # max candidate gaps to prior-art check per run
    PRIOR_ART_PER_SOURCE = int(env("PRIOR_ART_PER_SOURCE", 15))


settings = Settings()
