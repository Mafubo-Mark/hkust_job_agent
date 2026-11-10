"""Browser interface for the local HKUST job agent API."""
import os

import requests
import streamlit as st
import streamlit.components.v1 as components


API = os.getenv('HKUST_AGENT_API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='HKUST Career Assistant', page_icon='🎓', layout='centered')
st.title('HKUST Career Assistant')
st.caption('Upload a resume, choose HKUST job filters, then ask about your matches.')
st.session_state.setdefault('stage', 'upload')
st.session_state.setdefault('session_id', None)
st.session_state.setdefault('messages', [])
st.session_state.setdefault('error', '')


def api(method, path, **kwargs):
    """Call the local FastAPI server and turn response errors into readable messages."""
    try:
        response = requests.request(method, API + path, timeout=kwargs.pop('timeout', 30), **kwargs)
        if response.ok:
            return response.json()
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(str(detail))
    except requests.RequestException as error:
        raise RuntimeError('Cannot reach the local API. Start the app with python3 run_web.py.') from error


def restart():
    """Discard a pending session and return to the resume upload screen."""
    if st.session_state.session_id:
        try:
            api('DELETE', f'/sessions/{st.session_state.session_id}')
        except RuntimeError:
            pass
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.error = ''
    st.session_state.stage = 'upload'
    st.rerun()


@st.fragment(run_every='2s')
def watch_status():
    """Refresh the page when browser login or job preparation changes stage."""
    try:
        snapshot = api('GET', f'/sessions/{st.session_state.session_id}', timeout=5)
    except RuntimeError as error:
        st.error(str(error))
        return
    stage = snapshot['status']
    if stage in ('filters_ready', 'chat', 'error', 'closed'):
        st.session_state.snapshot = snapshot
        st.session_state.stage = stage
        st.rerun()
    if stage == 'waiting_login':
        st.info('Complete the HKUST login and declarations in the Chrome window that opened. Filter choices will appear here afterward.')
    else:
        st.info('Collecting filtered jobs, scoring them against your resume, and building the RAG index. This may take several minutes.')


def upload_screen():
    """Collect a searchable PDF and show a reminder below the upload controls."""
    st.subheader('1. Upload your resume')
    uploaded = st.file_uploader('Drag a searchable PDF here', type=['pdf'])
    if st.button('Continue to job filters', type='primary', disabled=uploaded is None):
        try:
            result = api('POST', '/sessions', files={'file': (uploaded.name, uploaded.getvalue(), 'application/pdf')}, timeout=60)
            st.session_state.session_id = result['session_id']
            st.session_state.stage = 'waiting_login'
            st.session_state.error = ''
            st.rerun()
        except RuntimeError as error:
            st.session_state.error = str(error)
    if st.session_state.error:
        st.error(st.session_state.error)
    st.caption('Resume reminder: PDF only, at most 20 MiB, with selectable text on every page. Scanned images and encrypted PDFs need re-exporting.')


def filters_screen():
    """Render the live HKUST dropdown choices as bounded frontend multiselects."""
    st.subheader('2. Choose job filters')
    st.caption('Leave any category empty to include all its options.')
    groups = st.session_state.snapshot['filter_options']
    with st.form('job_filters'):
        selections = {}
        for group in groups:
            labels = {item['value']: item['name'] for item in group['options']}
            selections[group['key']] = st.multiselect(
                f"{group['title']} (choose up to {group['limit']})",
                list(labels), format_func=lambda value, names=labels: names[value],
                key=f"filter_{group['key']}")
        submitted = st.form_submit_button('Find and score jobs', type='primary')
    if submitted:
        exceeded = [g['title'] for g in groups if len(selections[g['key']]) > g['limit']]
        if exceeded:
            st.error('Too many selections in: ' + ', '.join(exceeded))
        else:
            try:
                api('POST', f'/sessions/{st.session_state.session_id}/filters', json={'filters': selections})
                st.session_state.stage = 'preparing'
                st.rerun()
            except RuntimeError as error:
                st.error(str(error))
    st.button('Start over', on_click=restart)


def chat_screen():
    """Show the conversation and send each new message to the saved LangGraph graph."""
    st.subheader('3. Ask about your jobs')
    st.caption(f"{st.session_state.snapshot['job_count']} jobs scored. You can ask for top matches, requirements, resume advice, or interview preparation.")
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    question = st.chat_input('Ask a question, or say goodbye to end the session')
    if question:
        st.session_state.messages.append({'role': 'user', 'content': question})
        with st.spinner('Thinking...'):
            try:
                result = api('POST', f'/sessions/{st.session_state.session_id}/messages', json={'message': question}, timeout=180)
            except RuntimeError as error:
                result = {'answer': f'Could not finish this turn: {error}', 'end_session': False}
        st.session_state.messages.append({'role': 'assistant', 'content': result['answer']})
        if result['end_session']:
            st.session_state.session_id = None
            st.session_state.stage = 'closed'
        st.rerun()
    st.caption('Say “goodbye” or “that’s all, thanks” to end the session and delete its uploaded resume and generated files.')


if st.session_state.stage == 'upload':
    upload_screen()
elif st.session_state.stage in ('waiting_login', 'preparing'):
    st.subheader('Preparing your session')
    watch_status()
    if st.session_state.stage == 'waiting_login':
        st.button('Cancel and start over', on_click=restart)
elif st.session_state.stage == 'filters_ready':
    filters_screen()
elif st.session_state.stage == 'chat':
    chat_screen()
elif st.session_state.stage == 'error':
    st.error(st.session_state.snapshot.get('error') or 'Preparation failed.')
    st.button('Start over', on_click=restart)
else:
    st.success('Session ended. Your resume and generated files have been removed.')
    components.html('<script>try { window.parent.close(); } catch (e) {}</script>', height=0)
    st.caption('If your browser keeps this tab open, you can close it now.')
    st.button('Start a new session', on_click=restart)
