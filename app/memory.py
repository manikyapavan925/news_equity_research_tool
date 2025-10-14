"""
Simple JSONL-based memory persistence for article-derived memory.

Each record structure:
{
  "url": str,
  "title": str,
  "timestamp": str,  # ISO
  "memory": {
     "entities": [...],
     "reasons": [...],
     "sentiment": float,
     "titles": [...]
  }
}
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional


DEFAULT_PATH = os.path.join("data", "memory_store.jsonl")


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


class MemoryStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_PATH
        ensure_dir(self.path)
        # Create file if missing
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def add_record(self, record: Dict) -> None:
        ensure_dir(self.path)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_records(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def get_by_url(self, url: str, limit: int = 20) -> List[Dict]:
        results: List[Dict] = []
        for rec in self._iter_records():
            if rec.get("url") == url:
                results.append(rec)
        # Return latest first (assumes append-only chronological)
        return list(reversed(results))[:limit]

    def get_recent(self, limit: int = 50) -> List[Dict]:
        # Load all and take last N
        all_recs: List[Dict] = list(self._iter_records() or [])
        if not all_recs:
            return []
        return all_recs[-limit:][::-1]

    def get_recent_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        out: List[Dict] = []
        for rec in self._iter_records() or []:
            if rec.get('session_id') == session_id:
                out.append(rec)
        return out[-limit:][::-1]


def make_record_from_article_memory(url: str, title: str, timestamp: str, memory: Dict) -> Dict:
    return {
        "url": url,
        "title": title,
        "timestamp": timestamp,
        "memory": memory,
    }
