"""Start the local FastAPI and Streamlit services together."""
import subprocess
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent


def main():
    """Launch both servers and shut them down together on Ctrl+C."""
    backend = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api:app',
                                '--host', '127.0.0.1', '--port', '8000'], cwd=ROOT)
    frontend = None
    try:
        for _ in range(40):
            if backend.poll() is not None:
                raise RuntimeError('FastAPI stopped during startup.')
            try:
                if requests.get('http://127.0.0.1:8000/health', timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(.25)
        else:
            raise RuntimeError('FastAPI did not become ready on port 8000.')
        frontend = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
                                     '--server.address', '127.0.0.1', '--server.port', '8501'], cwd=ROOT)
        print('Open http://127.0.0.1:8501 if Streamlit does not open your browser automatically.')
        while frontend.poll() is None and backend.poll() is None:
            time.sleep(.5)
        if backend.poll() is not None:
            raise RuntimeError('FastAPI stopped; see its output above.')
    except KeyboardInterrupt:
        print('\nStopping the local web app.')
    finally:
        for process in (frontend, backend):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == '__main__':
    main()
