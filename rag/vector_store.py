"""Two local vector collections in SQLite; exact cosine search suits a personal job board."""
import json
import sqlite3
from contextlib import contextmanager
import numpy as np


class VectorStore:
    def __init__(self, path):
        """Create the vector database and its transactional metadata table."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS vectors (kind TEXT, job_id TEXT, source_index INT, text TEXT, vector TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')

    @contextmanager
    def connection(self):
        """Commit or roll back each SQLite operation and always close its connection."""
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def version(self):
        """Read the fingerprint of the source data used for the current index."""
        with self.connection() as db:
            row = db.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        return row[0] if row else None

    def replace(self, rows, vectors, version):
        """Commit both collections together only after all embeddings validate successfully."""
        if len(rows) != len(vectors):
            raise ValueError('Embedding count does not match document count.')
        if rows:
            array = np.asarray(vectors, dtype=float)
            if array.ndim != 2 or not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
                raise ValueError('Invalid embedding vectors.')
        with self.connection() as db:
            db.execute('DELETE FROM vectors')
            db.executemany('INSERT INTO vectors VALUES (?,?,?,?,?)',
                           [(*r, json.dumps(v)) for r, v in zip(rows, vectors)])
            db.execute("INSERT OR REPLACE INTO meta VALUES ('version',?)", (version,))

    def query(self, kind, vector, limit=None):
        """Rank distinct jobs by their best chunk cosine similarity, retaining source metadata."""
        with self.connection() as db:
            rows = db.execute('SELECT job_id,source_index,text,vector FROM vectors WHERE kind=?', (kind,)).fetchall()
        if not rows:
            return []
        matrix, query = np.asarray([json.loads(r[3]) for r in rows]), np.asarray(vector)
        if query.ndim != 1 or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError('Invalid query embedding.')
        scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        best = {}
        for row, score in zip(rows, scores):
            if row[0] not in best or score > best[row[0]]['similarity']:
                best[row[0]] = dict(job_id=row[0], source_index=row[1], text=row[2], similarity=float(score))
        return sorted(best.values(), key=lambda r: (-r['similarity'], r['job_id']))[:limit]
