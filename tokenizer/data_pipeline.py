"""Wikipedia data acquisition, cleaning, and corpus persistence.

The downloader uses MediaWiki's Action API. Article text is requested as a
plain-text extract; source metadata is retained for attribution and traceability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence
import unicodedata

import requests


@dataclass
class WikipediaArticle:
    """One Wikipedia article and the metadata needed to identify its source."""

    requested_title: str
    text: str
    canonical_title: Optional[str] = None
    page_id: Optional[int] = None
    revision_id: Optional[int] = None
    canonical_url: Optional[str] = None
    retrieved_at: Optional[str] = None
    license_name: Optional[str] = None
    source_metadata: Optional[dict[str, Any]] = None


class WikipediaArticleDownloader:
    """Download plain-text article extracts through the MediaWiki Action API."""

    def __init__(
        self,
        language: str = "en",
        user_agent: str = (
            "TransformerLearningProject/1.0 "
            "(educational project; https://github.com/Inialpha/GPT)"
        ),
        timeout: int = 30,
    ):
        if not language or not language.strip():
            raise ValueError("language must be a non-empty Wikipedia language code.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self.language = language.strip().lower()
        self.user_agent = user_agent
        self.timeout = timeout
        self.api_url = f"https://{self.language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": self.user_agent}

    def fetch_article(self, title: str) -> WikipediaArticle:
        """Fetch one article and return its text plus source metadata.

        Raises requests.RequestException for HTTP/connection failures,
        ValueError for invalid API responses, and LookupError when the title
        does not identify an existing article.
        """
        requested_title = title.strip() if isinstance(title, str) else ""
        if not requested_title:
            raise ValueError("title must be a non-empty string.")

        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "titles": requested_title,
            "prop": "extracts|info|revisions",
            "inprop": "url",
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "redirects": 1,
            "maxlag": 5,
        }

        # Respect Wikimedia's Retry-After guidance for temporary throttling
        # or service-unavailable responses.
        response = None
        for attempt in range(3):
            response = requests.get(
                self.api_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout,
            )
            if response.status_code not in (429, 503) or attempt == 2:
                break

            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 5 * (2 ** attempt)
            except ValueError:
                delay = 5 * (2 ** attempt)
            time.sleep(min(delay, 30))

        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                "Wikipedia returned a response that was not valid JSON."
            ) from exc

        if "error" in payload:
            error = payload["error"]
            raise ValueError(
                "Wikipedia API error "
                f"{error.get('code', 'unknown')}: "
                f"{error.get('info', 'No additional information.')}"
            )

        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            raise LookupError(f"Wikipedia returned no page for {requested_title!r}.")

        page = pages[0]
        if page.get("missing") or page.get("invalid"):
            raise LookupError(
                f"Wikipedia article {requested_title!r} does not exist or is invalid."
            )

        article_text = page.get("extract")
        if not isinstance(article_text, str) or not article_text.strip():
            raise ValueError(
                f"Wikipedia returned no article extract for {requested_title!r}."
            )

        revisions = page.get("revisions", [])
        latest_revision = revisions[0] if revisions else {}

        query = payload.get("query", {})
        source_metadata = {
            "namespace": page.get("ns"),
            "redirects": query.get("redirects", []),
            "normalized_titles": query.get("normalized", []),
        }

        return WikipediaArticle(
            requested_title=requested_title,
            canonical_title=page.get("title"),
            text=article_text,
            page_id=page.get("pageid"),
            revision_id=latest_revision.get("revid"),
            canonical_url=page.get("canonicalurl") or page.get("fullurl"),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            license_name="CC BY-SA 4.0 / GFDL; check page-specific notices",
            source_metadata=source_metadata,
        )

    def fetch_many(
        self,
        titles: Sequence[str],
        continue_on_error: bool = True,
    ) -> tuple[list[WikipediaArticle], dict[str, str]]:
        """Fetch several titles sequentially.

        TextExtracts only supports multiple extracts when exintro is enabled,
        which would return introductions rather than the full article text.
        Therefore this method deliberately calls fetch_article once per title.
        Requests are sequential to follow Wikimedia's API etiquette.

        Returns (successful_articles, failures), where failures maps the
        requested title to its error message.
        """
        articles: list[WikipediaArticle] = []
        failures: dict[str, str] = {}
        seen_titles: set[str] = set()

        for title in titles:
            normalized_title = title.strip() if isinstance(title, str) else ""
            if not normalized_title:
                failures[str(title)] = "Title must be a non-empty string."
                if not continue_on_error:
                    raise ValueError(failures[str(title)])
                continue

            title_key = normalized_title.casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            try:
                articles.append(self.fetch_article(normalized_title))
            except (requests.RequestException, ValueError, LookupError) as exc:
                failures[normalized_title] = str(exc)
                if not continue_on_error:
                    raise

        return articles, failures


class ArticleTextCleaner:
    """Clean plain-text extracts while preserving meaningful article prose."""

    def __init__(self, normalize_whitespace: bool = True):
        self.normalize_whitespace = normalize_whitespace

    def remove_markup(self, text: str) -> str:
        """Remove residual HTML markup and decode HTML entities.

        The downloader requests plaintext from TextExtracts, so this method
        is intentionally conservative; it is not a general wikitext parser.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = re.sub(r"(?is)<!--.*?-->", "", text)
        text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", "", text)
        text = re.sub(r"(?i)<br\s*/?>", "\n", text)
        text = re.sub(r"(?i)</(?:p|div|li|h[1-6])\s*>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return unescape(text)

    def normalize_text(self, text: str) -> str:
        """Normalize Unicode, control characters, spaces, and paragraph gaps."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        if self.normalize_whitespace:
            text = re.sub(r"[ \t\f\v]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def clean(self, text: str) -> str:
        """Apply markup removal followed by text normalization."""
        return self.normalize_text(self.remove_markup(text))


class CorpusBuilder:
    """Collect cleaned articles, assemble a corpus, and persist it to disk."""

    def __init__(self, separator: str = "\n\n"):
        self.separator = separator
        self._entries: list[tuple[WikipediaArticle, str]] = []
        self._seen_keys: set[str] = set()
        self._cached_text: Optional[str] = None
        self._cached_manifest: list[dict[str, Any]] = []

    @staticmethod
    def _article_key(article: WikipediaArticle) -> str:
        if article.page_id is not None:
            return f"page:{article.page_id}"
        title = article.canonical_title or article.requested_title
        return f"title:{title.strip().casefold()}"

    def add_article(self, article: WikipediaArticle, cleaned_text: str) -> None:
        """Add a cleaned article once, retaining its source metadata."""
        if not isinstance(article, WikipediaArticle):
            raise TypeError("article must be a WikipediaArticle instance.")
        if not isinstance(cleaned_text, str):
            raise TypeError("cleaned_text must be a string.")

        cleaned_text = cleaned_text.strip()
        if not cleaned_text:
            return

        key = self._article_key(article)
        if key in self._seen_keys:
            return

        self._seen_keys.add(key)
        self._entries.append((article, cleaned_text))

    def build(self) -> str:
        """Combine saved corpus text and newly added article text."""
        parts: list[str] = []
        if self._cached_text:
            parts.append(self._cached_text)
        parts.extend(text for _, text in self._entries)
        return self.separator.join(part for part in parts if part)

    def get_source_manifest(self) -> list[dict[str, Any]]:
        """Return provenance records for loaded and newly added articles."""
        manifest = list(self._cached_manifest)
        for article, _ in self._entries:
            record = asdict(article)
            # Article text is stored in the corpus file, not duplicated in
            # the manifest.
            record.pop("text", None)
            manifest.append(record)
        return manifest

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Save corpus text and a JSON source manifest.

        For example, saving to data/corpus.txt also creates
        data/corpus.manifest.json.
        """
        corpus_path = Path(path)
        if not corpus_path.suffix:
            corpus_path = corpus_path.with_suffix(".txt")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_path = corpus_path.with_suffix(".manifest.json")
        corpus_text = self.build()
        manifest = {
            "corpus_file": corpus_path.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "article_count": len(self.get_source_manifest()),
            "sources": self.get_source_manifest(),
        }

        corpus_path.write_text(corpus_text, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return corpus_path, manifest_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        separator: str = "\n\n",
    ) -> "CorpusBuilder":
        """Load a previously saved corpus without contacting Wikipedia."""
        corpus_path = Path(path)
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        builder = cls(separator=separator)
        builder._cached_text = corpus_path.read_text(encoding="utf-8")

        manifest_path = corpus_path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            builder._cached_manifest = manifest_data.get("sources", [])
            for source in builder._cached_manifest:
                page_id = source.get("page_id")
                title = source.get("canonical_title") or source.get("requested_title")
                if page_id is not None:
                    builder._seen_keys.add(f"page:{page_id}")
                elif title:
                    builder._seen_keys.add(f"title:{title.strip().casefold()}")

        return builder
