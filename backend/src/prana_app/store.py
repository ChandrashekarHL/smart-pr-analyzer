"""Persistence: webhook deliveries (idempotency), review runs and comments
(what the dashboard shows). SQLAlchemy 2; Postgres in production, SQLite
in tests - the same code."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Delivery(Base):
    __tablename__ = "deliveries"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)     # X-GitHub-Delivery
    event: Mapped[str] = mapped_column(String(64))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ReviewRun(Base):
    __tablename__ = "review_runs"
    __table_args__ = (UniqueConstraint("repo", "pr_number", "head_sha"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    installation_id: Mapped[int] = mapped_column(Integer)
    repo: Mapped[str] = mapped_column(String(200), index=True)
    pr_number: Mapped[int] = mapped_column(Integer)
    head_sha: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    hunks_total: Mapped[int] = mapped_column(Integer, default=0)
    hunks_reviewed: Mapped[int] = mapped_column(Integer, default=0)
    hunks_failed: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    review_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ReviewComment(Base):
    __tablename__ = "review_comments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("review_runs.id"), index=True)
    path: Mapped[str] = mapped_column(String(500))
    line: Mapped[int] = mapped_column(Integer)
    line_exact: Mapped[bool] = mapped_column(default=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    issue_type: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    inline: Mapped[bool] = mapped_column(default=True)     # False: held back in the summary
    fingerprint: Mapped[str] = mapped_column(String(16), index=True)
    summary: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    # Feedback (FR-10): 👍/👎 reactions on the posted comment, refreshed by the
    # worker's periodic `refresh_feedback` task (GitHub sends no reaction webhooks).
    github_comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    thumbs_up: Mapped[int] = mapped_column(Integer, default=0)
    thumbs_down: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Store:
    def __init__(self, url: str):
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, future=True, **kw)
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def record_delivery(self, delivery_id: str, event: str) -> bool:
        """True the first time a delivery id is seen; False for a redelivery."""
        with self.session() as s:
            s.add(Delivery(id=delivery_id, event=event))
            try:
                s.commit()
                return True
            except IntegrityError:
                s.rollback()
                return False

    def open_run(self, installation_id: int, repo: str, pr: int, head_sha: str) -> int | None:
        """A run id to enqueue, or None when this exact head is already queued,
        running or done. A failed run is reopened."""
        with self.session() as s:
            existing = s.scalar(select(ReviewRun).where(
                ReviewRun.repo == repo, ReviewRun.pr_number == pr,
                ReviewRun.head_sha == head_sha))
            if existing is not None:
                if existing.status != "failed":
                    return None
                existing.status, existing.error = "pending", None
                s.commit()
                return existing.id
            run = ReviewRun(installation_id=installation_id, repo=repo, pr_number=pr,
                            head_sha=head_sha)
            s.add(run)
            try:
                s.commit()
            except IntegrityError:          # a concurrent delivery won the race
                s.rollback()
                return None
            return run.id

    def get_run(self, run_id: int) -> ReviewRun | None:
        with self.session() as s:
            return s.get(ReviewRun, run_id)

    def mark(self, run_id: int, status: str, error: str | None = None) -> None:
        with self.session() as s:
            run = s.get(ReviewRun, run_id)
            run.status = status
            if status == "processing":
                run.attempts += 1
            if error is not None:
                run.error = error[:2000]
            if status in ("failed", "superseded"):
                run.finished_at = now()
            s.commit()

    def finish(self, run_id: int, outcome) -> None:
        with self.session() as s:
            run = s.get(ReviewRun, run_id)
            run.status = outcome.status
            run.finished_at = now()
            run.seconds = round(outcome.seconds, 3)
            run.hunks_total = outcome.hunks_total
            run.hunks_reviewed = outcome.hunks_reviewed
            run.hunks_failed = outcome.hunks_failed
            run.duplicates = outcome.duplicates
            run.review_id = outcome.review_id
            run.error = None
            for f, inline in [(f, True) for f in outcome.posted] + \
                             [(f, False) for f in outcome.low_confidence]:
                s.add(ReviewComment(run_id=run_id, path=f.path, line=f.line,
                                    line_exact=f.line_exact, severity=f.severity,
                                    issue_type=f.issue_type, confidence=f.confidence,
                                    inline=inline, fingerprint=f.fingerprint,
                                    summary=" ".join(f.explanation.split())[:300]))
            s.commit()

    # ------------------------------------------------------------- feedback

    def feedback_targets(self, days: int = 7) -> list[tuple[int, str, int]]:
        """(installation, repo, PR) with comments posted in the last `days`."""
        since = now() - timedelta(days=days)
        with self.session() as s:
            rows = s.execute(select(ReviewRun.installation_id, ReviewRun.repo,
                                    ReviewRun.pr_number)
                             .where(ReviewRun.status == "posted",
                                    ReviewRun.finished_at >= since)
                             .distinct()).all()
            return [tuple(r) for r in rows]

    def update_feedback(self, repo: str, pr: int,
                        items: list[tuple[str, int, int, int]]) -> int:
        """items: (fingerprint, github comment id, 👍, 👎). Returns rows updated."""
        if not items:
            return 0
        by_fp = {fp: (cid, up, down) for fp, cid, up, down in items}
        n = 0
        with self.session() as s:
            q = (select(ReviewComment).join(ReviewRun, ReviewComment.run_id == ReviewRun.id)
                 .where(ReviewRun.repo == repo, ReviewRun.pr_number == pr,
                        ReviewComment.inline, ReviewComment.fingerprint.in_(list(by_fp))))
            for c in s.scalars(q):
                c.github_comment_id, c.thumbs_up, c.thumbs_down = by_fp[c.fingerprint]
                n += 1
            s.commit()
        return n

    # ------------------------------------------------------------ dashboard

    def recent_runs(self, limit: int = 50) -> list[dict]:
        with self.session() as s:
            runs = s.scalars(select(ReviewRun).order_by(ReviewRun.id.desc()).limit(limit)).all()
            return [{c: getattr(r, c) for c in (
                "id", "repo", "pr_number", "head_sha", "status", "attempts", "created_at",
                "finished_at", "seconds", "hunks_total", "hunks_reviewed", "hunks_failed",
                "duplicates", "error")} for r in runs]

    def stats(self) -> dict:
        with self.session() as s:
            by_status = dict(s.execute(select(ReviewRun.status, func.count())
                                       .group_by(ReviewRun.status)).all())
            by_severity = dict(s.execute(select(ReviewComment.severity, func.count())
                                         .where(ReviewComment.inline)
                                         .group_by(ReviewComment.severity)).all())
            secs = sorted(x for x in s.scalars(select(ReviewRun.seconds)
                                               .where(ReviewRun.seconds.is_not(None))))
            return {"runs_by_status": by_status, "comments_by_severity": by_severity,
                    "review_seconds_p50": percentile(secs, 0.5),
                    "review_seconds_p95": percentile(secs, 0.95), "reviews_timed": len(secs)}

    def dashboard(self, days: int = 30, recent: int = 50) -> dict:
        """Everything the dashboard shows, in one call (FR-10)."""
        since = now() - timedelta(days=days)
        with self.session() as s:
            runs = s.scalars(select(ReviewRun).where(ReviewRun.created_at >= since)).all()
            comments = s.scalars(select(ReviewComment).join(
                ReviewRun, ReviewComment.run_id == ReviewRun.id)
                .where(ReviewRun.created_at >= since)).all()
            repo_of = {r.id: (r.repo, r.pr_number) for r in runs}

        inline = [c for c in comments if c.inline]
        daily: dict[str, list[float]] = {}
        count_by_day: dict[str, int] = {}
        for r in runs:
            day = _day(r.created_at)
            count_by_day[day] = count_by_day.get(day, 0) + 1
            if r.seconds is not None:
                daily.setdefault(day, []).append(r.seconds)
        secs = sorted(r.seconds for r in runs if r.seconds is not None)
        up = sum(c.thumbs_up for c in inline)
        down = sum(c.thumbs_down for c in inline)
        rated = sum(1 for c in inline if c.thumbs_up or c.thumbs_down)
        bins = [0] * 10
        for c in inline:
            bins[min(int(c.confidence * 10), 9)] += 1
        return {
            "window_days": days,
            "totals": {"runs": len(runs),
                       "reviews_posted": sum(r.status.startswith("posted") for r in runs),
                       "failed": sum(r.status == "failed" for r in runs),
                       "comments_inline": len(inline),
                       "comments_held_back": len(comments) - len(inline),
                       "hunks_reviewed": sum(r.hunks_reviewed for r in runs)},
            "runs_by_status": _count(r.status for r in runs),
            "comments_by_severity": _count(c.severity for c in inline),
            "comments_by_issue_type": _count(c.issue_type for c in inline),
            "confidence_histogram": [{"bin": f"{i / 10:.1f}–{(i + 1) / 10:.1f}", "comments": n}
                                     for i, n in enumerate(bins)],
            "feedback": {"thumbs_up": up, "thumbs_down": down, "rated_comments": rated,
                         "helpful_rate": round(up / (up + down), 3) if up + down else None},
            "latency": {"p50_s": percentile(secs, 0.5), "p95_s": percentile(secs, 0.95),
                        "timed_runs": len(secs)},
            "daily": [{"day": d, "runs": count_by_day[d],
                       "p50_s": percentile(sorted(daily.get(d, [])), 0.5),
                       "p95_s": percentile(sorted(daily.get(d, [])), 0.95)}
                      for d in sorted(count_by_day)],
            "recent_runs": [{"id": r.id, "repo": r.repo, "pr": r.pr_number,
                             "status": r.status, "created_at": r.created_at.isoformat(),
                             "seconds": r.seconds, "hunks": f"{r.hunks_reviewed}/{r.hunks_total}",
                             "error": (r.error or "")[:200]}
                            for r in sorted(runs, key=lambda r: r.id, reverse=True)[:recent]],
            "recent_comments": [{"repo": repo_of[c.run_id][0], "pr": repo_of[c.run_id][1],
                                 "path": c.path, "line": c.line, "severity": c.severity,
                                 "issue_type": c.issue_type,
                                 "confidence": round(c.confidence, 3),
                                 "thumbs_up": c.thumbs_up, "thumbs_down": c.thumbs_down,
                                 "summary": c.summary}
                                for c in sorted(inline, key=lambda c: c.id, reverse=True)[:recent]],
        }


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an already sorted list; None when empty."""
    if not sorted_values:
        return None
    k = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return round(sorted_values[k], 3)


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _day(dt: datetime) -> str:
    if dt.tzinfo is None:                   # SQLite returns naive datetimes (stored as UTC)
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%d")
