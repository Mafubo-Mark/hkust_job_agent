# Implementation and verification report

Date: 22 September 2026

The project now implements the PRD's preparation workflow, all five named tools, DeepSeek-controlled specialist routing, local semantic job RAG, and persistent multi-turn dialogue through LangGraph. Source files contain a docstring for every Python function. The compact matcher was reused rather than retaining the older duplicated 1,660-line implementation.

## What was built

| Requirement | Implementation |
|---|---|
| Searchable PDF only; no OCR | Existing pypdf reader validates pages and fingerprints the original PDF. Blank/image-only and encrypted PDFs are rejected before scraping. |
| Resume summary | `tool_summarize` copies text up to 500 units; otherwise DeepSeek summarizes with a checked limit. Original PDF text is still used for scoring, avoiding reliance on a compressed resume for score evidence. |
| Existing scraping and filtering | `job_matcher_tool.py` retains manual HKUST login, dropdown filters, pagination, concurrent authenticated browser detail requests, deadline checks, and normalization. |
| Three job outputs | `data/jobs.json`, `data/jobs_nodes.json`, and `data/result.json`; the middle file contains only company/title. The latest PRD filename replaces `jobs_no_des.json`. |
| Scores and rankings | Existing evidence-based scoring checks source quotes and requirement coverage. Python calculates weighted scores and deterministic rankings. Unchanged job/resume/model versions reuse valid cached results. |
| Dual job RAG | `rag/` creates separate name and full-description collections. Sentence Transformers produces multilingual embeddings; SQLite stores vectors and NumPy performs cosine retrieval. Full descriptions are chunked with overlap. |
| Agent-chosen tools | DeepSeek plans specialist handoffs and each specialist uses native function calls. Tool argument schemas and per-specialist allowlists are enforced in Python. |
| Multi-turn conversation | LangGraph SQLite checkpoints preserve message history, displayed job order, selected job ID, and company. A changed dataset clears outdated conversation references. |
| Transparent outputs | Tool-use labels come from executed calls. Ranking fields and web source links are formatted by Python. |

## The five tools

1. **`tool_summarize`** — copy or summarize resume text to at most 500 counted units. Preparation writes `resume.txt`; chat summarization is read-only so it cannot silently invalidate scored data.
2. **`tool_return_jobs`** — return a 1-based inclusive ranking range with company, title, Ref No., score, and eligibility. It saves the exact displayed job order for subsequent references.
3. **`tool_job_resume_question`** — resolve an ID, Ref No., displayed ordinal, remembered pronoun, or mixed company/title phrase. Feed the complete matching job, saved assessment, and resume summary to DeepSeek for fit, resume edits, requirements, or interview practice. Ambiguous names produce clarification candidates.
4. **`tool_find_required_jobs`** — retrieve all indexed jobs in semantic order and evaluate every job against the requested criteria. Return matches, source quotations, and uncertain cases. This deliberately costs one classification request per job; a small top-k cutoff would not fulfill “all matching jobs.”
5. **`tool_search`** — call Baidu Qianfan web search, filter unusable URLs, and summarize returned snippets with source links. Scope checking happens before the search request. This tool does not pretend that model memory is a web search.

## How the files work together

`main.py` loads `config.py`, runs `workflows/preparation_graph.py` when a resume is supplied, then opens `workflows/dialogue_graph.py` with a SQLite checkpointer whose connection stays open for the session.

Preparation follows `extract → summarize → match → index`. It stages results in a temporary directory. If extraction, AI scoring, or index construction fails, the existing prepared dataset is retained. The readiness marker is published last. It detects incomplete publication and modified files; it is not a guarantee of rollback after a disk failure during final publication.

For each dialogue turn, `agents/supervisor.py` uses DeepSeek to choose up to three specialist tasks. `job_agent.py`, `resume_agent.py`, and `company_search_agent.py` share a compact implementation in `specialist.py`, but have separate prompts and tool allowlists. They choose tools through native function calling. `tools/job_tools.py` validates arguments and invokes the five tool modules. Tool results return to the requesting specialist through the graph, which may request another tool or finish. Eight model decisions bound a turn.

`rag/document_loader.py` checks that title-only rows match full-job source order; stable job IDs are stored alongside source indices. `embeddings.py` runs the local semantic model. `vector_store.py` commits both collections transactionally. `retriever.py` checks freshness, searches the appropriate collection, and resolves references. No cloud vector database is required.

The README includes the complete directory tree, a workflow diagram, installation commands, and example dialogue.

## Verification performed

The automated suite uses the **real LangGraph graphs, SQLite checkpointer, SQLite vector store, NumPy similarity, pypdf extraction, and scoring/validation code**. Tests inject scripted model replies, synthetic job data, and mocked browser/search responses. They verify orchestration and request contracts; they are not evidence that a live model will always choose correctly.

| Test module | Tests | What was checked |
|---|---:|---|
| `test_preparation_graph.py` | 4 | Full and saved-job preparation; filter forwarding; Chrome cleanup; invalid PDFs stop before scraping; failed scoring preserves the previous dataset. |
| `test_matcher.py` | 7 | Real PDF text extraction and fingerprint; blank/encrypted rejection; original scoring and cache reuse; fabricated quote/omitted requirement rejection; reference parsing; standalone three-output contract; atomic JSON failure handling. |
| `test_rag.py` | 7 | Both collections cover all job IDs; unchanged indexes skip embedding; stale data and misaligned title files are rejected; exact/ordinal/pronoun/mixed-name resolution; ambiguity; empty datasets; reindex cannot approve an edited resume. |
| `test_tools.py` | 9 | Ranking ranges and fields; strict arguments; multilingual summary counting; summary limit failures; saved resume evidence; every job examined for requirements; unsupported quotes rejected; web scope/source handling; Qianfan request contract. |
| `test_dialogue_graph.py` | 7 | Top jobs → second job → resume edits → company search; persistence after reopening and in a separate Python process; thread isolation; new dataset resets references; scope refusal; multiple specialists; bounded repeated calls; invalid calls and tool allowlists. |
| `test_providers.py` | 2 | DeepSeek native function-call schema, JSON mode, disabled thinking setting, connection cleanup, and rejection of truncated replies. |
| **Total** | **36** | **All passed.** |

Additional checks:

- All Python files compiled successfully; `main.py --help` worked.
- An AST audit found **zero Python functions without an explanatory docstring**.
- The **real** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model was downloaded and exercised, without a paid API. It returned 384-dimensional vectors. “华为人工智能研究员” versus “Huawei AI Researcher” had cosine similarity **0.812**, compared with **0.294** against “Bank accounting intern.” This is a multilingual smoke test, not a retrieval benchmark.
- Automated tests do not need network access or real credentials; model downloads occur only when using the production semantic embedder.
- The verified files were installed in the requested `hkust_job_agent` folder. All 36 tests passed again from that folder using its `.venv`, and `pip check` reported no broken requirements. The virtual environment reuses existing system packages and installs the missing checkpoint dependencies locally. The embedding weights are cached in `data/model_cache/`.

The original project files were backed up before replacement at `/Users/mafubo/Documents/Codex/2026-09-20/e/work/agent_original`. Raw test output from the installed project is saved in `reports/test_results.txt`.

Run the suite with `python -m unittest discover -s tests -v` after installing requirements.

## What still requires a live run

No live HKUST authentication/scrape, paid DeepSeek scoring/dialogue, or authenticated Qianfan search was performed during this implementation. These require the user's configured credentials, real searchable resume, and institutional login. The code for those integrations is present and request shapes were checked against current official documentation. Live provider account access, page selectors, output quality, latency, and cost remain unverified.

No actual job rankings or resume files were fabricated to make the folder look prepared. Populate `.env`, then run `python main.py --resume "/path/to/resume.pdf"`. First-time model loading needs internet; subsequent embedding uses cached weights. Use `--score-saved` to avoid another scrape when only the resume changes.

Current limitations are explicit: this is a local single-user terminal app; conversation IDs isolate chats but share one prepared resume/job dataset. Semantic name resolution and model classification are probabilistic. Full criteria searches can be slow because every job is examined. Resume/job scoring sends evidence to DeepSeek. Career-centre information in this PRD uses web search, not an additional local document library.

## API references consulted

- [LangGraph SQLite checkpointer](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/README.md) — context-managed checkpointer lifetime.
- [DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls/) and [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/) — native tools and provider request fields.
- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/) — explicitly disabled for the compact tool-calling loop.
- [Baidu Qianfan web search](https://cloud.baidu.com/doc/qianfan-api/s/Wmbq4z7e5) — search endpoint, authentication options, query constraints, and returned references.
