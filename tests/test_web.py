"""Local API checks for PDF validation, filter choices, and farewell cleanup."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api
from tests.test_matcher import pdf_bytes


class WebSessionTests(unittest.TestCase):
    def test_upload_filter_chat_and_goodbye_remove_private_session(self):
        """One browser session selects a live filter and deletes its resume on farewell."""
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(api, 'SESSIONS_DIR', Path(directory) / 'sessions'), \
             patch.object(api.Session, 'start'):
            with TestClient(api.app) as client:
                invalid = client.post('/sessions', files={'file': ('resume.pdf', b'not a PDF', 'application/pdf')})
                self.assertEqual(invalid.status_code, 400)
                created = client.post('/sessions', files={'file': ('resume.pdf', pdf_bytes(), 'application/pdf')})
                self.assertEqual(created.status_code, 200)
                session_id = created.json()['session_id']
                session = api.sessions[session_id]
                self.assertTrue(session.resume_path.exists())
                session.options = [{'key': 'EM[]', 'title': 'Employment Mode', 'limit': 2,
                                    'options': [{'name': 'Full Time', 'value': 'full'}]}]
                session.status = 'filters_ready'
                bad = client.post(f'/sessions/{session_id}/filters', json={'filters': {'EM[]': ['unknown']}})
                self.assertEqual(bad.status_code, 422)
                selected = client.post(f'/sessions/{session_id}/filters', json={'filters': {'EM[]': ['full']}})
                self.assertEqual(selected.status_code, 200)
                self.assertEqual(session.filter_queue.get_nowait(), {'EM[]': ['full']})
                session.status = 'chat'
                session.app = Mock()
                session.app.invoke.return_value = {'answer': 'Goodbye.', 'end_session': True, 'turn_count': 1}
                ended = client.post(f'/sessions/{session_id}/messages', json={'message': 'goodbye'})
                self.assertEqual(ended.status_code, 200)
                self.assertTrue(ended.json()['end_session'])
                self.assertFalse(session.ctx.data.exists())
                self.assertEqual(client.get(f'/sessions/{session_id}').status_code, 404)
