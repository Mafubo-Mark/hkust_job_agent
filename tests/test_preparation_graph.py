import unittest
from unittest.mock import patch
from tests.helpers import fixture
from workflows.preparation_graph import prepare
import job_matcher_tool as matcher


PROFILE = {'status': 'extracted', 'resume_version': 'a' * 64, 'text': 'Built a Python project.',
           'pages': [{'page': 1, 'text': 'Built a Python project.'}]}


class PreparationTests(unittest.TestCase):
    def test_saved_pipeline_outputs_and_no_browser(self):
        """The real preparation graph publishes four required data files and both vector collections."""
        ctx = fixture(self)
        with patch.object(matcher, 'read_searchable_resume', return_value=PROFILE), \
             patch.object(matcher, 'legacy_scores', return_value=ctx.rankings()), \
             patch.object(matcher, 'create_driver') as browser:
            result = prepare(ctx, '/synthetic.pdf', mode='score_saved')
        browser.assert_not_called()
        self.assertEqual(result['count'], 3)
        self.assertEqual(ctx.resume(), PROFILE['text'])
        self.assertEqual(ctx.ready(), ctx.dataset_id())
        self.assertFalse(list(ctx.data.glob('.preparing-*')))
        self.assertTrue((ctx.data / 'jobs_nodes.json').exists())

    def test_scoring_failure_keeps_old_outputs(self):
        """Failed external scoring preserves all previously prepared files and their readiness marker."""
        ctx = fixture(self)
        before = ctx.dataset_id()
        with patch.object(matcher, 'read_searchable_resume', return_value=PROFILE), \
             patch.object(matcher, 'legacy_scores', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaises(RuntimeError):
                prepare(ctx, '/synthetic.pdf', mode='score_saved')
        self.assertEqual(ctx.ready(), before)
        self.assertFalse(list(ctx.data.glob('.preparing-*')))

    def test_invalid_pdf_prevents_browser_and_scoring(self):
        """Rejected image-only PDFs never reach scraping or paid scoring."""
        ctx = fixture(self)
        with patch.object(matcher, 'read_searchable_resume', side_effect=ValueError('image-only')), \
             patch.object(matcher, 'create_driver') as browser, patch.object(matcher, 'legacy_scores') as score:
            with self.assertRaises(ValueError):
                prepare(ctx, '/scan.pdf')
        browser.assert_not_called()
        score.assert_not_called()

    def test_full_preparation_closes_browser(self):
        """A successful scrape passes filters and always closes Chrome before scoring/indexing."""
        ctx = fixture(self)
        with patch.object(matcher, 'read_searchable_resume', return_value=PROFILE), \
             patch.object(matcher, 'create_driver') as browser, \
             patch.object(matcher, 'collect_jobs', return_value=[matcher.JobRecord(**j) for j in ctx.jobs()]) as collect, \
             patch.object(matcher, 'legacy_scores', return_value=ctx.rankings()):
            prepare(ctx, '/synthetic.pdf', filters={})
        self.assertEqual(collect.call_args.args[2], {})
        browser.return_value.quit.assert_called_once()
