import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import job_matcher_tool as m
from tests.test_matcher import JOB, REPLY
from tests.test_preparation_graph import PROFILE
from tests.helpers import fixture
from workflows.preparation_graph import prepare


class RecoveryTests(unittest.TestCase):
    def test_duplicate_key_gets_specific_correction(self):
        """Reject duplicate job_quote keys and send the specific error to a successful retry."""
        duplicate = REPLY.replace('"job_quote":', '"job_quote":"duplicate", "job_quote":', 1)
        client = Mock()
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=reply))])
            for reply in (duplicate, REPLY)]
        progress = Mock()
        with patch('openai.OpenAI', return_value=client):
            rows = m.score_jobs([m.JobRecord(**JOB)], m.ResumeProfile(**PROFILE),
                               m.Settings(api_key='synthetic', model='mock'), [], on_progress=progress)
        self.assertEqual(rows[0]['score'], 100)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertIn('Duplicate JSON key: job_quote', client.chat.completions.create.call_args.kwargs['messages'][-1]['content'])
        progress.assert_called_once_with(rows)

    def test_bounded_failures_preserve_completed_progress(self):
        """Four invalid responses fail safely without discarding the first completed score."""
        first = m.JobRecord(**JOB)
        second = first.model_copy(update={'job_id': 'b'})
        client = Mock()
        client.chat.completions.create.side_effect = [SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=reply))])
            for reply in (REPLY, 'bad', 'bad', 'bad', 'bad')]
        progress = Mock()
        with patch('openai.OpenAI', return_value=client), self.assertRaises(ValueError):
            m.score_jobs([first, second], m.ResumeProfile(**PROFILE),
                         m.Settings(api_key='synthetic', model='mock'), [], on_progress=progress)
        self.assertEqual(client.chat.completions.create.call_count, 5)
        progress.assert_called_once()
        self.assertEqual(progress.call_args.args[0][0]['job_id'], first.job_id)

    def test_resume_checkpoint_without_browser(self):
        """Retain jobs and partial scores on failure, then resume and publish without scraping."""
        ctx = fixture(self)
        old, rows = ctx.dataset_id(), ctx.rankings()

        def fail_after_one(jobs, resume, settings, previous, force, on_progress):
            """Emulate a scoring failure after one score is checkpointed."""
            on_progress(rows[:1])
            raise ValueError('Duplicate JSON key: job_quote')

        with patch.object(m, 'read_searchable_resume', return_value=PROFILE), \
             patch.object(m, 'create_driver'), \
             patch.object(m, 'collect_jobs', return_value=[m.JobRecord(**j) for j in ctx.jobs()]), \
             patch.object(m, 'legacy_scores', side_effect=fail_after_one), self.assertRaises(ValueError):
            prepare(ctx, '/synthetic.pdf')
        pending = m.read_json(ctx.data / 'pending_match.json')
        self.assertEqual(len(pending['jobs']), 3)
        self.assertEqual(len(pending['results']), 1)
        self.assertEqual(ctx.ready(), old)
        with patch.object(m, 'read_searchable_resume', return_value=PROFILE), \
             patch.object(m, 'create_driver') as browser, \
             patch.object(m, 'legacy_scores', return_value=rows) as scorer:
            prepare(ctx, '/synthetic.pdf', mode='resume_scoring')
        browser.assert_not_called()
        self.assertEqual(scorer.call_args.args[3], rows[:1])
        self.assertFalse((ctx.data / 'pending_match.json').exists())
        self.assertEqual(ctx.ready(), ctx.dataset_id())
