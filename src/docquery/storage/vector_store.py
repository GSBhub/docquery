import hashlib
import heapq
import logging
import math
import sqlite3
import struct
from itertools import islice
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from docquery.config import Settings

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    def __init__(self, db_path: str, embedding_dim: int):
        self._db_path = db_path
        self._dim = embedding_dim
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _drop_vec0_if_present(self) -> None:
        """Drop the old sqlite-vec virtual table if this database was created with the previous schema."""
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'"
        ).fetchone()
        if row is None or "vec0" not in (row[0] or ""):
            return
        try:
            self._conn.execute("DROP TABLE chunk_embeddings")
            self._conn.commit()
            logger.info("Dropped legacy vec0 chunk_embeddings table; will recreate as BLOB table")
        except sqlite3.OperationalError:
            raise RuntimeError(
                f"Database '{self._db_path}' uses the old sqlite-vec schema and cannot be "
                f"migrated automatically (the vec0 module is no longer installed).\n"
                f"Delete the file and re-ingest:\n"
                f"  rm {self._db_path}\n"
                f"  docquery ingest <your.pdf> --db {self._db_path}"
            )

    def _init_schema(self) -> None:
        self._drop_vec0_if_present()
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                hash        TEXT UNIQUE,
                content     TEXT,
                source      TEXT,
                page        INTEGER,
                chunk_index INTEGER
            );

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                rowid     INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(content, source, content='chunks', content_rowid='id');
        """)
        self._conn.commit()

    def add_chunks(self, chunks: list[Document], embeddings: Embeddings, batch_size: int = 32) -> tuple[int, int]:
        batches = list(_batched(chunks, batch_size))
        all_embeddings: list[list[float]] = []

        for batch in tqdm(batches, desc="Embedding chunks", unit="batch"):
            texts = [c.page_content for c in batch]
            vecs = embeddings.embed_documents(texts)
            all_embeddings.extend(vecs)
            logger.debug("Embedded batch of %d chunks", len(texts))

        inserted = 0
        skipped = 0
        cur = self._conn.cursor()

        for chunk, vec in tqdm(zip(chunks, all_embeddings), desc="Writing to DB", unit="chunk", total=len(chunks)):
            h = _sha256(chunk.page_content)
            try:
                cur.execute(
                    "INSERT INTO chunks (hash, content, source, page, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    (h, chunk.page_content, chunk.metadata.get("source", ""), chunk.metadata.get("page", 0), chunk.metadata.get("chunk_index", 0)),
                )
                rowid = cur.lastrowid
                embedding_blob = struct.pack(f"{len(vec)}d", *vec)
                cur.execute(
                    "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
                    (rowid, embedding_blob),
                )
                cur.execute(
                    "INSERT INTO chunks_fts (rowid, content, source) VALUES (?, ?, ?)",
                    (rowid, chunk.page_content, chunk.metadata.get("source", "")),
                )
                inserted += 1
                logger.debug("Inserted chunk rowid=%d", rowid)
            except sqlite3.IntegrityError:
                skipped += 1
                logger.debug("Skipped duplicate chunk hash=%s", h[:8])

        self._conn.commit()
        logger.info("Inserted %d chunks, skipped %d duplicates", inserted, skipped)
        return inserted, skipped

    def similarity_search(self, query_vec: list[float], k: int = 5) -> list[dict[str, Any]]:
        logger.debug("similarity_search k=%d vec_dim=%d", k, len(query_vec))
        rows = self._conn.execute("SELECT rowid, embedding FROM chunk_embeddings").fetchall()
        if not rows:
            return []

        scored = []
        fmt = f"{self._dim}d"
        for row in rows:
            vec = list(struct.unpack(fmt, row["embedding"]))
            sim = _cosine_sim(query_vec, vec)
            scored.append((sim, row["rowid"]))

        top = heapq.nlargest(k, scored, key=lambda x: x[0])
        top_rowids = [rowid for _, rowid in top]
        sim_by_rowid = {rowid: sim for sim, rowid in top}

        placeholders = ",".join("?" * len(top_rowids))
        chunk_rows = self._conn.execute(
            f"SELECT id, content, source, page, chunk_index FROM chunks WHERE id IN ({placeholders})",
            top_rowids,
        ).fetchall()

        results = sorted(
            [dict(r) | {"similarity": sim_by_rowid[r["id"]]} for r in chunk_rows],
            key=lambda r: r["similarity"],
            reverse=True,
        )
        logger.debug("similarity_search returned %d results", len(results))
        return results

    def keyword_search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        logger.debug("FTS5 keyword_search query=%r", query)
        phrase = '"' + query.replace('"', ' ') + '"'
        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.content, c.source, c.page, c.chunk_index
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (phrase, k),
            ).fetchall()
        except Exception:
            logger.debug("FTS5 phrase search failed, falling back to LIKE")
            rows = self._conn.execute(
                "SELECT id, content, source, page, chunk_index FROM chunks WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", k),
            ).fetchall()
        return [dict(r) for r in rows]

    def page_lookup(self, page_num: int) -> list[dict[str, Any]]:
        logger.debug("page_lookup page=%d", page_num)
        rows = self._conn.execute(
            "SELECT id, content, source, page, chunk_index FROM chunks WHERE page = ? ORDER BY chunk_index",
            (page_num,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
