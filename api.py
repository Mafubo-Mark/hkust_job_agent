"""Local FastAPI service for resume upload, HKUST filters, and LangGraph chat."""
import queue
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, field_validator

import job_matcher_tool as matcher
from config import Context
from diagnostics import describe_error, report_error
from workflows.dialogue_graph import build_dialogue_graph
from workflows.preparation_graph import prepare


ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / 'data' / 'web_sessions'
sessions = {}
sessions_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    """Remove local web-session resumes when the API server stops."""
    yield
    with sessions_lock:
        active = list(sessions.values())
        sessions.clear()
    for session in active:
        session.close()
    shutil.rmtree(SESSIONS_DIR, ignore_errors=True)


app = FastAPI(title='HKUST Job Agent', docs_url=None, redoc_url=None, lifespan=lifespan)


class FilterRequest(BaseModel):
    """Carry selected HKUST option values from the Streamlit form."""
    filters: dict[str, list[str]]


class ChatRequest(BaseModel):
    """Carry one user message to the saved LangGraph session."""
    message: str = Field(min_length=1, max_length=12000)

    @field_validator('message')
    @classmethod
    def nonempty_message(cls, value):
        """Reject whitespace-only chat turns before they reach LangGraph."""
        if not value.strip():
            raise ValueError('Enter a question or a goodbye message.')
        return value.strip()


class Session:
    """Own one resume, browser login, filter choices, and in-memory conversation."""
    def __init__(self, directory):
        self.id = uuid.uuid4().hex
        self.ctx = Context(ROOT, data_dir=directory)
        self.resume_path = self.ctx.data / 'resume.pdf'
        self.status = 'waiting_login'
        self.error = None
        self.options = []
        self.count = 0
        self.app = None
        self.lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.filter_queue = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.worker = None

    def start(self):
        """Open Chrome in a background thread while the page polls for login completion."""
        self.worker = threading.Thread(target=self._prepare, daemon=True)
        self.worker.start()

    def _prepare(self):
        """Read live filter options, then reuse the logged-in browser for preparation."""
        driver = None
        try:
            driver = matcher.create_driver()
            matcher.login_to_hkust(driver, self.ctx.settings)
            if self.closed.is_set():
                return
            options = []
            for name, title, limit in matcher.FILTERS:
                select = driver.find_element(matcher.By.NAME, name)
                items = driver.execute_script("""return [...arguments[0].options]
                    .filter(o => o.value && !o.text.includes('Clear All'))
                    .map(o => ({value: o.value, name: o.text.trim()}));""", select)
                options.append({'key': name, 'title': title, 'limit': limit, 'options': items})
            with self.lock:
                self.options, self.status = options, 'filters_ready'
            filters = self.filter_queue.get(timeout=1800)
            if filters is None or self.closed.is_set():
                return
            with self.lock:
                self.status = 'preparing'
            self.ctx.driver = driver
            result = prepare(self.ctx, str(self.resume_path), 'full', filters, True)
            self.ctx.driver = None
            driver.quit()
            driver = None
            self.ctx.ready()
            graph = build_dialogue_graph(self.ctx, InMemorySaver())
            if not self.closed.is_set():
                with self.lock:
                    self.app, self.count, self.status = graph, result['count'], 'chat'
        except Exception as error:
            if not self.closed.is_set():
                report_error(error, self.ctx)
                with self.lock:
                    self.error, self.status = describe_error(error, self.ctx), 'error'
        finally:
            self.ctx.driver = None
            if driver is not None:
                driver.quit()
            if self.closed.is_set():
                self._dispose()

    def snapshot(self):
        """Return only UI state, without resume text, credentials, or model objects."""
        with self.lock:
            return {'session_id': self.id, 'status': self.status, 'error': self.error,
                    'filter_options': self.options if self.status == 'filters_ready' else [],
                    'job_count': self.count}

    def submit_filters(self, filters):
        """Validate selected values and wake the browser preparation worker."""
        with self.lock:
            if self.status != 'filters_ready':
                raise HTTPException(409, 'Filters are not ready or were already submitted.')
            allowed = {group['key']: group for group in self.options}
            if set(filters) - allowed.keys():
                raise HTTPException(422, 'Unknown filter category.')
            for key, values in filters.items():
                choices = {item['value'] for item in allowed[key]['options']}
                if (len(values) > allowed[key]['limit'] or len(set(values)) != len(values)
                        or not set(values).issubset(choices)):
                    raise HTTPException(422, f'Invalid selection for {allowed[key]["title"]}.')
            self.status = 'preparing'
        self.filter_queue.put_nowait(filters)

    def ask(self, message):
        """Run one turn against this session's LangGraph memory and close on farewell."""
        with self.chat_lock:
            if self.status != 'chat' or self.app is None:
                raise HTTPException(409, 'Chat is not ready or has ended.')
            try:
                state = self.app.invoke({'messages': [HumanMessage(content=message)]},
                                        {'configurable': {'thread_id': self.id}, 'recursion_limit': 64})
            except Exception as error:
                report_error(error, self.ctx)
                raise HTTPException(500, describe_error(error, self.ctx)) from error
            answer = {'answer': state['answer'], 'end_session': bool(state.get('end_session')),
                      'turn_count': state.get('turn_count', 0)}
            if answer['end_session']:
                self.close()
            return answer

    def _dispose(self):
        """Close services and remove the uploaded PDF plus all session-generated files."""
        client = self.ctx._llm
        if client is not None and hasattr(client, 'close'):
            client.close()
        self.ctx._llm, self.app = None, None
        shutil.rmtree(self.ctx.data, ignore_errors=True)

    def close(self):
        """End this session and interrupt a pending filter-selection wait."""
        with self.lock:
            if self.closed.is_set():
                return
            self.closed.set()
            self.status = 'closed'
        try:
            self.filter_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker is None or not self.worker.is_alive():
            self._dispose()


def get_session(session_id):
    """Find a live session without exposing any other user's data."""
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, 'Session not found.')
    return session


@app.get('/health')
def health():
    """Let the launcher and frontend confirm that the local API is available."""
    return {'status': 'ok'}


@app.post('/sessions')
def create_session(file: UploadFile = File(...)):
    """Validate an uploaded searchable PDF before opening the HKUST browser."""
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(400, 'Please upload a searchable PDF resume.')
    content = file.file.read(matcher.MAX_BYTES + 1)
    if len(content) > matcher.MAX_BYTES:
        raise HTTPException(400, 'Resume exceeds the 20 MiB file limit.')
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session = Session(tempfile.mkdtemp(prefix='web-', dir=SESSIONS_DIR))
    try:
        session.resume_path.write_bytes(content)
        matcher.read_searchable_resume(session.resume_path)
    except (OSError, ValueError) as error:
        session.close()
        raise HTTPException(400, str(error)) from error
    with sessions_lock:
        sessions[session.id] = session
    session.start()
    return session.snapshot()


@app.get('/sessions/{session_id}')
def session_status(session_id: str):
    """Report login, filter, and preparation progress to Streamlit."""
    return get_session(session_id).snapshot()


@app.post('/sessions/{session_id}/filters')
def submit_filters(session_id: str, request: FilterRequest):
    """Send frontend filter selections to the waiting preparation worker."""
    session = get_session(session_id)
    session.submit_filters(request.filters)
    return session.snapshot()


@app.post('/sessions/{session_id}/messages')
def send_message(session_id: str, request: ChatRequest):
    """Answer one user turn and discard session data when LangGraph ends it."""
    session = get_session(session_id)
    result = session.ask(request.message.strip())
    if result['end_session']:
        with sessions_lock:
            sessions.pop(session_id, None)
    return result


@app.delete('/sessions/{session_id}')
def delete_session(session_id: str):
    """Allow the UI to abandon an upload, filter wait, or failed preparation."""
    session = get_session(session_id)
    session.close()
    with sessions_lock:
        sessions.pop(session_id, None)
    return {'status': 'closed'}
