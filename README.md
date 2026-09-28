# 🎤 Meeting-to-Action-Items AI

Turn a meeting transcript (or audio recording) into a summary, key discussion points, decisions, action items with owners, deadlines and priorities, and a downloadable report. Built with Streamlit and runs on free, open-source NLP models. **No API key is needed.**

## Features

- **Two input types:** paste or upload a `.txt` transcript, or upload audio (`.wav`, `.mp3`, `.m4a`)
- **Summary:** abstractive summarization with DistilBART
- **Entity extraction:** people, organizations and dates via spaCy NER, plus tech and project keywords
- **Key points, decisions and action items:** extracted with NLP and rule-based cues
- **Speaker identification:** reads `Name: text` labels in the transcript
- **Priority detection:** High / Medium / Low, based on keywords such as "urgent", "asap" and "eventually"
- **Dashboard:** editable action-items table (status and priority), charts by person and status, speaker participation, deadlines
- **Email drafts:** one draft per assigned person
- **Meeting history:** past meetings are stored in SQLite; reopen or delete them
- **Export:** report as DOCX or PDF, tasks as CSV (imports into Trello, Asana, Jira, ClickUp and Notion)

## Project structure

```
your-repo/
├── app.py             # the entire application
├── requirements.txt   # Python dependencies
└── packages.txt       # system packages (ffmpeg, needed for audio)
```

### requirements.txt

```
streamlit>=1.32
numpy<2
spacy>=3.7
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
transformers>=4.38,<5
torch>=2.2
sentencepiece
pandas
matplotlib
python-docx
fpdf2
SpeechRecognition
pydub
```

(`deep-translator` is no longer needed. Translation was removed.)

### packages.txt

```
ffmpeg
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For audio uploads locally, install ffmpeg on your machine (`sudo apt install ffmpeg` or `brew install ffmpeg`).

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `packages.txt` to a GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo and branch, and set the main file to `app.py`.
4. Deploy. No secrets or API keys are required.
5. **After every code change, push to GitHub and let the app redeploy.** Editing a file locally does not update the live app.

The first run is slow because it downloads the DistilBART model (and the DistilBERT QA model if it is used). After that the models are cached.

## How to use

1. Open **New Meeting**, enter a title, and paste or upload a transcript (or upload audio and click **Transcribe audio**).
2. Prefix each line with the speaker's name (`Priya: I will prepare the test cases by Thursday.`). This greatly improves who gets assigned each task.
3. Click **Analyze Meeting**.
4. Review results in **Dashboard**, edit statuses in the table, and browse old meetings in **History**.
5. Download reports and task CSVs from **Export**.

## Example transcript

```
Dharshini: There's a critical bug in production and it needs to be fixed immediately.
Dharshini: I will fix the login bug today since it's urgent.
Priya: I will prepare the test cases. I should have them ready by Thursday.
Bob: I will deploy the homepage changes to the server by Friday.
Rahul: The analytics dashboard is low priority, I'll finish it eventually.
Dharshini: We agreed to go with the current homepage layout, so that's approved.
```

## Known limitations

- **Extraction is rule-based, not an LLM.** Action items are picked up from cues like "will", "should", "needs to" and "please", so it can miss tasks or flag ordinary sentences. It works best on clear, explicit statements. Review and edit the table.
- **Memory:** torch plus the transformer models are heavy for Streamlit Cloud's free tier (about 1 GB RAM). If the app crashes while analyzing, use a smaller model or a host with more memory.
- **Audio transcription needs internet.** It uses Google's free web speech recognition in 55-second chunks, so it works best for short, clear recordings and does not separate speakers. Add `Name:` labels to the transcribed text before analyzing.
- **History is not permanent on Streamlit Cloud.** The disk is ephemeral, so `meetings.db` resets when the app restarts or redeploys. For permanent storage, swap the SQLite functions for a hosted database such as Supabase or Postgres.
- **PDF export is Latin-1 only.** Non-Latin characters are replaced with `?`. Use the DOCX export if you need other scripts.
- **Q&A and transcript search are in the code but not shown in the UI.** `answer_question()` and `search_transcript()` exist in `app.py` but no page calls them yet. They can be added as a new tab.
- **Translation was removed.** The free Google Translate endpoint kept hitting its rate limit.

## Tech stack

Streamlit · spaCy · Hugging Face Transformers (DistilBART, DistilBERT-SQuAD) · PyTorch · pandas · matplotlib · python-docx · fpdf2 · SpeechRecognition · pydub · SQLite
