# Original scoring workflow restored — 23 September 2026

The source reference was `/Users/mafubo/Desktop/All Files/其他/Project/hkust_job_project/`: `hkust_job_scraper.py`, `job_matcher.py`, and `main.py`. Those original files were left unchanged.

The original scraper supplies the existing manual login, dropdown filtering, deadline exclusion, pagination, and concurrent browser detail-download approach. Those behaviors remain in the agent's portable matcher. The agent stores normalized full job records for its tools instead of the old scraper's legacy wrapper fields.

The agent's preparation graph and standalone matcher now use the original matcher's **two-line scoring method**. Its prompt was compared programmatically with `job_matcher.py` and matches exactly. DeepSeek receives the full extracted resume and complete normalized job description; Python parses `overall score`, prints each raw response, ranks descending, and checkpoints after each success. The configurable DeepSeek model and `.env` credentials remain in use.

`result.json` includes both the original `Company/Organization`, `Job Title/Job Nature`, and `deepseek result` fields and the agent's `company`, `title`, `job_id`, `ref_no`, numeric `score`, and `rank`. Resume/job/model/method fingerprints support safe cache reuse. Two-line responses do not establish eligibility, so it is labeled unknown. Resume advice is generated separately using job and resume content.

Unlike the original script, the agent does not silently skip a failed score and present a partial ranking as complete. Invalid scores receive up to three attempts, and failed runs can resume from `data/pending_match.json`. Existing published results remain intact until preparation completes. Old structured-evidence scores are not reused as original-method scores.

The old source folder is not independently runnable as supplied: its files import `config`, `storage`, and `tools` modules absent from that folder. Its `main.py` also references a collection entry point absent from the supplied scraper. The relevant behavior was therefore integrated into the working agent rather than importing that incomplete folder at runtime.

Verification: **44 automated tests passed**, including the real preparation and dialogue graphs, checkpoint recovery, original plain-text scoring, raw-response output fields, descending ranking, per-success saving, cache reuse/invalidation, and rejection of invalid scores. The original prompt equality check passed. No paid API call or new scrape was performed for this change.

This update supersedes the structured-scoring description in the earlier implementation report. The `score_jobs` evidence scorer remains available for compatibility and regression tests, but the normal agent preparation path now calls `legacy_scores`, which implements the original two-line method.
