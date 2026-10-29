"""The post step: publish one GitHub review, or write it to disk in dry-run mode."""

from datetime import datetime
from pathlib import Path

from .models import ReviewReport
from .report import inline_comments, review_event


def post_review(ctx, report: ReviewReport, body: str) -> str:
    """Post the review on the PR and return its URL."""
    comments = inline_comments(report.findings, {f.path: f.patch for f in ctx.files})
    return ctx.create_review(body=body, event=review_event(report), comments=comments)


def write_review(ctx, report: ReviewReport, body: str, out_dir: Path = Path("results")) -> str:
    """Dry-run stand-in for post_review: save the review that would have been posted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{ctx.repo_full_name.replace('/', '_')}_pr{ctx.number}_{datetime.now():%Y%m%d-%H%M%S}"
    path = out_dir / f"{slug}.md"
    comments = inline_comments(report.findings, {f.path: f.patch for f in ctx.files})
    lines = [body, "", "---", f"Event: {review_event(report)}", f"Inline comments: {len(comments)}"]
    lines += [f"- {c['path']}:{c['line']}" for c in comments]
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"{slug}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return str(path)
