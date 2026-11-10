import unittest
from job_matcher_tool import read_json, write_json
from rag.build_index import build_index, mark_ready
from rag.retriever import retrieve, resolve_job
from tests.helpers import fixture, ScriptedLLM


class RagTests(unittest.TestCase):
    def test_two_indexes_and_stable_ids(self):
        """Both indexes return original IDs and all jobs, including chunked descriptions."""
        ctx = fixture(self, count=12)
        for kind in ('names', 'jobs'):
            rows = retrieve(ctx, 'Huawei AI', kind, None)
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['job_id'] for r in rows}, {f'job-{i}' for i in range(12)})

    def test_index_reuse_and_stale_detection(self):
        """Unchanged sources skip embedding and changed sources cannot query a stale index."""
        ctx = fixture(self)
        calls = ctx.embedder.calls
        build_index(ctx)
        self.assertEqual(ctx.embedder.calls, calls)
        rows = read_json(ctx.data / 'jobs.json')
        rows[0]['description'] = 'Changed requirements'
        write_json(ctx.data / 'jobs.json', rows)
        with self.assertRaises(ValueError):
            retrieve(ctx, 'Python')
        build_index(ctx)
        self.assertGreater(ctx.embedder.calls, calls)

    def test_misaligned_names_rejected(self):
        """A reordered title-only file cannot silently map a title to the wrong job."""
        ctx = fixture(self)
        write_json(ctx.data / 'jobs_nodes.json', list(reversed(read_json(ctx.data / 'jobs_nodes.json'))))
        with self.assertRaises(ValueError):
            build_index(ctx)

    def test_ordinal_pronoun_and_reference(self):
        """Ordinals use displayed order, pronouns use selection, and Ref No. is exact."""
        ctx = fixture(self)
        state = {'displayed_job_ids': ['job-2', 'job-0'], 'selected_job_id': 'job-1'}
        self.assertEqual(resolve_job(ctx, 'the second one', state)['job']['job_id'], 'job-0')
        self.assertEqual(resolve_job(ctx, 'it', state)['job']['job_id'], 'job-1')
        self.assertEqual(resolve_job(ctx, 'REF-2', {})['job']['job_id'], 'job-2')
        self.assertEqual(resolve_job(ctx, 'it', {})['status'], 'ambiguous')
        self.assertEqual(resolve_job(ctx, 'the fifth one', state)['status'], 'not_found')

    def test_mixed_name_and_ambiguity(self):
        """Semantic candidates feed disambiguation, and unrecognized IDs are never accepted."""
        llm = ScriptedLLM(objects=[{'job_id': 'job-1', 'ambiguous': False},
                                  {'job_id': 'fake', 'ambiguous': False}])
        ctx = fixture(self, llm)
        self.assertEqual(resolve_job(ctx, 'huawei ai research', {})['job']['job_id'], 'job-1')
        self.assertEqual(resolve_job(ctx, 'Huawei', {})['status'], 'ambiguous')

    def test_empty_index(self):
        """An empty filtered board remains a valid, searchable empty dataset."""
        ctx = fixture(self, count=0)
        self.assertEqual(retrieve(ctx, 'Python'), [])

    def test_reindex_cannot_bless_a_changed_resume(self):
        """Changing resume.txt requires new scoring, not just a fresh index marker."""
        ctx = fixture(self)
        (ctx.data / 'resume.txt').write_text('Unscored resume changes')
        with self.assertRaises(ValueError):
            mark_ready(ctx)
