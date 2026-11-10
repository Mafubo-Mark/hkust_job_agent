import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import job_matcher_tool as m
from tests.test_matcher import JOB
from tests.test_preparation_graph import PROFILE


def reply(text):
    """Represent a plain-text DeepSeek completion without structured evidence JSON."""
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text))])


class OriginalScoringTests(unittest.TestCase):
    def setUp(self):
        """Provide synthetic job/resume inputs and a test-only provider configuration."""
        self.job = m.JobRecord(**JOB)
        self.resume = m.ResumeProfile(**PROFILE)
        self.settings = m.Settings(api_key='synthetic', model='mock')

    def test_plain_scores_rank_and_save_after_each_success(self):
        """Parse original two-line answers, keep raw responses, rank jobs, and checkpoint each success."""
        client = Mock()
        client.chat.completions.create.side_effect = [reply('overall score: 70/100\nrecommendation level: Apply with targeted resume revision'),
                                                     reply('overall score: 90/100\nrecommendation level: Strongly recommend applying')]
        snapshots = []

        def save(rows):
            """Capture independent copies of each progress callback for assertions."""
            snapshots.append(copy.deepcopy(rows))

        with patch('openai.OpenAI', return_value=client):
            rows = m.legacy_scores([self.job, self.job.model_copy(update={'job_id': 'b'})],
                                   self.resume, self.settings, on_progress=save)
        self.assertEqual([r['score'] for r in rows], [90, 70])
        self.assertEqual([r['rank'] for r in rows], [1, 2])
        self.assertEqual([len(r) for r in snapshots], [1, 2])
        self.assertEqual(rows[0]['deepseek result'], rows[0]['response'])
        self.assertEqual(rows[0]['eligibility'], 'unknown')
        self.assertNotIn('response_format', client.chat.completions.create.call_args.kwargs)
        self.assertIn('EXACTLY two lines', client.chat.completions.create.call_args.kwargs['messages'][0]['content'])
        client.close.assert_called_once()

    def test_original_cache_reused_but_old_evidence_cache_is_not(self):
        """Reuse only scores from this method and matching resume/job/model versions."""
        client = Mock()
        client.chat.completions.create.return_value = reply('overall score: 80/100\nrecommendation level: Apply with targeted resume revision')
        with patch('openai.OpenAI', return_value=client):
            rows = m.legacy_scores([self.job], self.resume, self.settings)
        with patch('openai.OpenAI') as api:
            self.assertEqual(m.legacy_scores([self.job], self.resume, self.settings, rows), rows)
        api.assert_not_called()
        rows[0]['scoring_method'] = 'different-method'
        client.reset_mock()
        with patch('openai.OpenAI', return_value=client):
            m.legacy_scores([self.job], self.resume, self.settings, rows)
        client.chat.completions.create.assert_called_once()

    def test_invalid_score_does_not_publish_or_checkpoint(self):
        """Malformed scores cannot silently become -1, zero, or a completed ranking."""
        client, progress = Mock(), Mock()
        client.chat.completions.create.return_value = reply('No numeric score')
        with patch('openai.OpenAI', return_value=client), self.assertRaises(ValueError):
            m.legacy_scores([self.job], self.resume, self.settings, on_progress=progress)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        progress.assert_not_called()
        client.close.assert_called_once()
