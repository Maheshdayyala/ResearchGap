from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class Search(db.Model):
    __tablename__ = "searches"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    query = db.Column(db.String(300), nullable=False)
    status = db.Column(db.String(20), default="queued")   # queued|running|retrieved|done|failed
    stage = db.Column(db.String(30), default="queued")
    progress = db.Column(db.Integer, default=0)
    message = db.Column(db.String(400), default="")
    error = db.Column(db.Text)
    meta = db.Column(db.JSON, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    papers = db.relationship("Paper", backref="search", cascade="all, delete-orphan")
    report = db.relationship("GapReport", backref="search", uselist=False, cascade="all, delete-orphan")


class Paper(db.Model):
    __tablename__ = "papers"
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, index=True)
    pid = db.Column(db.String(20), index=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.JSON, default=list)
    year = db.Column(db.Integer)
    abstract = db.Column(db.Text, default="")
    venue = db.Column(db.String(300))
    doi = db.Column(db.String(200))
    url = db.Column(db.String(500))
    citations_count = db.Column(db.Integer, default=0)
    rank_score = db.Column(db.Float, default=0.0)
    sources = db.Column(db.JSON, default=list)
    external_ids = db.Column(db.JSON, default=dict)
    references = db.Column(db.JSON, default=list)
    structured = db.Column(db.JSON)

    def to_dict(self):
        return {"db_id": self.id, "pid": self.pid, "title": self.title, "authors": self.authors or [], "year": self.year,
                "abstract": self.abstract or "", "venue": self.venue or "", "doi": self.doi, "url": self.url or "",
                "citations": self.citations_count or 0, "rank_score": self.rank_score or 0, "sources": self.sources or [],
                "external_ids": self.external_ids or {}, "references": self.references or [], "structured": self.structured}


class Citation(db.Model):
    __tablename__ = "citations"
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, index=True)
    citing_pid = db.Column(db.String(20), nullable=False)
    cited_pid = db.Column(db.String(20), nullable=False)


class GapReport(db.Model):
    __tablename__ = "gap_reports"
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    candidates = db.Column(db.JSON, default=list)       # detected / evaluated gaps (evaluated ones carry "novelty")
    verified_gap = db.Column(db.JSON)
    novelty = db.Column(db.JSON)
    plan = db.Column(db.JSON)
    report_md = db.Column(db.Text, default="")
    status = db.Column(db.String(20), default="draft")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    """Conversation messages scoped to one user's saved analysis."""
    __tablename__ = "chat_messages"
    id = db.Column(db.Integer, primary_key=True)
    search_id = db.Column(db.Integer, db.ForeignKey("searches.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    role = db.Column(db.String(16), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
