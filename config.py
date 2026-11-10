"""Shared configuration, validated data access, and injectable external services."""
import hashlib
import json
import os
from pathlib import Path
from dotenv import dotenv_values
from job_matcher_tool import JobRecord, job_fingerprint, load_settings, read_json


class Context:
    def __init__(self, root, llm=None, embedder=None, search=None, data_dir=None):
        """Keep configuration and services outside LangGraph's serializable state."""
        self.root = Path(root).resolve()
        self.data = Path(data_dir).resolve() if data_dir else self.root / 'data'
        self.data.mkdir(parents=True, exist_ok=True) #creates that folder if it doesn’t already exist.
        self.settings = load_settings(self.root)
        self.env = {**dotenv_values(self.root / '.env', interpolate=False), **os.environ}
        self._llm, self._embedder, self.search = llm, embedder, search

    @property
    def llm(self):
        """Create the DeepSeek client only when a model call is needed."""
        if self._llm is None:
            from agents.llm import DeepSeek
            self._llm = DeepSeek(self.settings)
        return self._llm

    @property
    def embedder(self):
        """Load a local multilingual embedding model lazily, never at import time."""
        if self._embedder is None:
            from rag.embeddings import Embeddings
            self._embedder = Embeddings(self.env.get('EMBEDDING_MODEL') or
                                       'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                                       self.root / 'data' / 'model_cache')
        return self._embedder

    def jobs(self):
        """Validate saved full job records and enforce unique stable identifiers."""
        rows = read_json(self.data / 'jobs.json')
        rows = [JobRecord.model_validate(row).model_dump(mode='json') for row in rows]
        if len({row['job_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate job IDs in jobs.json.')
        return rows

    def rankings(self):
        """Load validated scores, using saved rank order rather than model-generated order."""
        rows = read_json(self.data / 'result.json')
        jobs = {j['job_id']: j for j in self.jobs()}
        if not isinstance(rows, list) or len(rows) != len(jobs):
            raise ValueError('Run preparation: every saved job needs a score.')
        if len({r.get('job_id') for r in rows}) != len(rows):
            raise ValueError('Duplicate scored jobs.')
        for r in rows:
            if r.get('job_id') not in jobs or type(r.get('score')) not in (int, float) or not 0 <= r['score'] <= 100:
                raise ValueError('Invalid score or unknown job in result.json.')
            if r.get('job_version') and r['job_version'] != job_fingerprint(JobRecord(**jobs[r['job_id']])):
                raise ValueError('A job changed since scoring; start a fresh run with your PDF.')
        versions = {r.get('resume_version') for r in rows}
        if len(versions) > 1:
            raise ValueError('Results mix different resume versions; rescore all jobs.')
        ordered = sorted(rows, key=lambda r: (r.get('rank', float('inf')), -r['score'], r['job_id']))
        return [{**r, **jobs[r['job_id']], 'rank': i + 1} for i, r in enumerate(ordered)]

    def resume(self):
        """Read the prepared resume; never accept an LLM-invented resume as evidence."""
        text = (self.data / 'resume.txt').read_text(encoding='utf-8').strip()
        if not text:
            raise ValueError('The prepared resume is empty.')
        return text

    def dataset_id(self):
        """Fingerprint files so a changed resume or ranking invalidates old job references."""
        digest = hashlib.sha256()
        for name in ('jobs.json', 'jobs_nodes.json', 'result.json', 'resume.txt'):
            digest.update((self.data / name).read_bytes())
        return digest.hexdigest()

    def ready(self):
        """Reject incomplete preparation and manual edits until the index is rebuilt."""
        manifest = read_json(self.data / 'prepared.json')
        if manifest.get('dataset_id') != self.dataset_id():
            raise ValueError('Prepared files changed; start a fresh run with your PDF.')
        self.rankings()
        self.resume()
        return manifest['dataset_id']


def packed(value):
    """Serialize source data as JSON, with Unicode preserved for multilingual prompts."""
    return json.dumps(value, ensure_ascii=False)
