"""Render repository Markdown links that live outside the Sphinx source tree."""

from pathlib import Path
import os
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit


_LINK = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)\s]+)(?P<suffix>[^)]*)\)")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,}|:{3,})")


def _without_inline_code(line):
    """Mask code spans without changing offsets into the original line."""
    masked = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        end = index
        while end < len(line) and line[end] == "`":
            end += 1
        marker = line[index:end]
        close = line.find(marker, end)
        if close < 0:
            index = end
            continue
        for position in range(index, close + len(marker)):
            masked[position] = " "
        index = close + len(marker)
    return "".join(masked)


def rewrite_repo_links(
    text, source_path, docs_root, repo_root, revision, repository="Jittor/jittor"
):
    """Link existing repo files outside docs to the same revision on GitHub.

    Links inside docs remain MyST cross-references. A missing repository file
    fails the build instead of becoming an unchecked external URL.
    """
    source_path = Path(source_path).resolve()
    docs_root = Path(docs_root).resolve()
    repo_root = Path(repo_root).resolve()
    output = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            kind = marker.group(1)[0]
            if fence is None:
                fence = kind
            elif fence == kind:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue
        mask = _without_inline_code(line)
        for match in reversed(list(_LINK.finditer(mask))):
            raw = match.group("target")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
                continue
            candidate = (source_path.parent / unquote(parsed.path)).resolve()
            try:
                relative = candidate.relative_to(repo_root)
            except ValueError:
                continue
            try:
                candidate.relative_to(docs_root)
            except ValueError:
                pass
            else:
                continue
            if not candidate.is_file():
                raise FileNotFoundError("missing repository link target: {}".format(relative))
            url_path = "/{}/blob/{}/{}".format(repository, revision, quote(relative.as_posix()))
            url = urlunsplit(("https", "github.com", url_path, parsed.query, parsed.fragment))
            line = line[: match.start("target")] + url + line[match.end("target") :]
        output.append(line)
    return "".join(output)


def setup(app):
    """Resolve repository files as links without weakening MyST's docs links."""
    docs_root = Path(__file__).resolve().parent
    repo_root = docs_root.parent
    repository = os.environ.get("GITHUB_REPOSITORY", "Jittor/jittor")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid GITHUB_REPOSITORY: {}".format(repository))
    revision = os.environ.get("GITHUB_SHA", "2.0-refactor")
    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|2\.0-refactor)", revision):
        raise ValueError("invalid GITHUB_SHA: {}".format(revision))

    def on_source_read(app, docname, source):
        path = Path(app.env.doc2path(docname))
        source[0] = rewrite_repo_links(
            source[0], path, docs_root, repo_root, revision, repository
        )

    app.connect("source-read", on_source_read)
    return {"parallel_read_safe": True}
