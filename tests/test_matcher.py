"""Regression tests for the retained scraper, searchable-PDF reader, and evidence scorer."""
import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from selenium.common.exceptions import ElementClickInterceptedException
import job_matcher_tool as m


def pdf_bytes(text='Built a Python project.', encrypted=False):
    """Build a real tiny PDF fixture with searchable text, a blank page, or encryption."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text is not None:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('test-password')
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


JOB = {'job_id': 'a', 'company': 'Example', 'title': 'Developer', 'description': 'Python experience required.',
       'requirements': ['Python experience'], 'url': 'https://example.com/job/1'}
REPLY = json.dumps({'evidence': [{'requirement': 'Python experience', 'category': 'skills', 'mandatory': True,
    'requirement_index': 0, 'job_quote': 'Python experience required.', 'outcome': 'match',
    'explanation': 'Python project evidence.', 'resume_page': 1, 'resume_quote': 'Built a Python project.'}]})


class MatcherTests(unittest.TestCase):
    def setUp(self):
        """Create a real searchable PDF and a private output directory for every test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pdf = self.root / 'resume.pdf'
        self.pdf.write_bytes(pdf_bytes())
        self.profile = m.ResumeProfile.model_validate(m.read_searchable_resume(self.pdf))
        self.job = m.JobRecord(**JOB)
        self.settings = m.Settings(api_key='synthetic-test-token', model='mock-model')

    def test_real_pdf_extraction_and_fingerprint(self):
        """pypdf extracts real text and fingerprints the PDF bytes for score-cache invalidation."""
        self.assertEqual(self.profile.text, 'Built a Python project.')
        self.assertEqual(self.profile.resume_version, hashlib.sha256(self.pdf.read_bytes()).hexdigest())

    def test_blank_and_encrypted_pdf_rejected(self):
        """Image-only/blank and encrypted PDFs are rejected without OCR."""
        for text, encrypted in [(None, False), ('Some searchable text', True)]:
            self.pdf.write_bytes(pdf_bytes(text, encrypted))
            with self.assertRaises(ValueError):
                m.read_searchable_resume(self.pdf)

    def test_evidence_scoring_and_cache(self):
        """Real scoring code computes 100 from cited evidence and skips unchanged cached jobs."""
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=REPLY))])
        with patch('openai.OpenAI', return_value=client):
            rows = m.score_jobs([self.job], self.profile, self.settings, [], False)
        self.assertEqual(rows[0]['score'], 100)
        self.assertEqual(rows[0]['eligibility'], 'eligible')
        with patch('openai.OpenAI') as api:
            cached = m.score_jobs([self.job], self.profile, self.settings, rows, False)
        api.assert_not_called()
        self.assertEqual(cached, rows)

    def test_invented_quote_and_missing_coverage_rejected(self):
        """The scorer rejects invented resume evidence and omitted requirements."""
        reply = json.loads(REPLY)
        reply['evidence'][0]['resume_quote'] = 'Invented PhD'
        item = m.build_assessment(json.dumps(reply), self.job, self.profile, 'mock', 'v1')
        with self.assertRaises(ValueError):
            m.validate_assessment(item, self.job, self.profile)
        job = self.job.model_copy(update={'requirements': ['Python experience', 'Java experience']})
        with self.assertRaises(ValueError):
            m.build_assessment(REPLY, job, self.profile, 'mock', 'v1')

    def test_collection_uses_ref_and_normalization(self):
        """The retained collection pipeline parses Ref No. and normalized requirements."""
        row = {'company': 'Example', 'job_title_nature': 'Developer', 'application_deadline': 'Until filled',
               'url': 'https://example.com/job/1'}
        batches = [[{'url': row['url'], 'ok': True, 'text': 'Ref No.: DEMO-1\nRequirements\nPython experience required.'}]]
        with patch.object(m, 'login_to_hkust'), patch.object(m, 'filter_jobs'), \
             patch.object(m, 'extract_current_page_jobs', return_value=[row]), \
             patch.object(m, 'browser_fetch_details', return_value=(batch for batch in batches)), \
             patch.object(m, 'click_next_page', return_value=False):
            rows = m.collect_jobs(Mock(), self.settings, {})
        self.assertEqual(rows[0].ref_no, 'DEMO-1')
        self.assertIn('Python experience required.', rows[0].requirements)

    def test_filter_search_click_works_when_page_overlay_intercepts_it(self):
        """A covering filter overlay cannot prevent the search form from submitting."""
        driver = Mock()
        submit = driver.find_element.return_value
        submit.click.side_effect = ElementClickInterceptedException('covered')
        driver.execute_script.return_value = []
        with patch.object(m, 'WebDriverWait'), patch.object(m.time, 'sleep'):
            m.filter_jobs(driver, {})
        driver.execute_script.assert_any_call('arguments[0].click();', submit)

    def test_standalone_tool_three_outputs(self):
        """The portable matcher still works independently and uses the PRD's three JSON filenames."""
        with patch.object(m, 'create_driver'), patch.object(m, 'collect_jobs', return_value=[self.job]):
            result = m.job_matcher(str(self.pdf), str(self.root / 'out'), 'prepare', {})
        self.assertEqual(result['count'], 1)
        self.assertEqual({p.name for p in (self.root / 'out').iterdir()}, {'jobs.json', 'jobs_nodes.json', 'result.json'})

    def test_failed_atomic_write_preserves_file(self):
        """A failed JSON replacement leaves the old file and removes temporary siblings."""
        target = self.root / 'result.json'
        m.write_json(target, ['old'])
        with patch.object(Path, 'replace', side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError):
                m.write_json(target, ['new'])
        self.assertEqual(m.read_json(target), ['old'])
        self.assertFalse(list(self.root.glob('*.tmp')))
