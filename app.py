"""
Meeting-to-Action-Items AI  —  single-file Streamlit app.

Upload a meeting AUDIO file OR a TRANSCRIPT and get: summary, key discussion
points, decisions, action items (person/deadline/priority), entities,
charts, search, Q&A, translation, PDF export, email drafts, CSV/JSON/Trello
task export, and multi-meeting history (SQLite).

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then deploy on https://share.streamlit.io
pointing at app.py. Set GEMINI_API_KEY in Streamlit Cloud's
Settings > Secrets, or just paste it in the sidebar at runtime.

NOTE ON GEMINI: this uses Google's OpenAI-compatibility endpoint, so the
`openai` Python package still works -- we just point it at Google's URL
and use Gemini model names instead of GPT model names. Get a Gemini key
(free tier available) at https://aistudio.google.com/apikey
"""

import io
import re
import csv
import json
import sqlite3
import tempfile
import datetime
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from openai import OpenAI
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from deep_translator import GoogleTranslator


# ============================================================================
# SECTION 1: DATABASE (meeting history, SQLite)
# ----------------------------------------------------------------------------
# NOTE on Streamlit Community Cloud: the filesystem there is EPHEMERAL — it
# resets on redeploy/sleep. This SQLite file is fine for demos and a single
# active session. For real persistence, swap this for hosted Postgres
# (e.g. Supabase/Railway) — only this section would need to change.
# ============================================================================

DB_PATH = Path(__file__).parent / "meetings.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT,
            transcript TEXT,
            data_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_meeting(title: str, transcript: str, data: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO meetings (title, created_at, transcript, data_json) VALUES (?, ?, ?, ?)",
        (title, datetime.datetime.now().isoformat(timespec="seconds"), transcript, json.dumps(data)),
    )
    conn.commit()
    meeting_id = cur.lastrowid
    conn.close()
    return meeting_id


def update_meeting_data(meeting_id: int, data: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE meetings SET data_json = ? WHERE id = ?", (json.dumps(data), meeting_id))
    conn.commit()
    conn.close()


def get_all_meetings() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, title, created_at FROM meetings ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_meeting(meeting_id: int) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    result["data"] = json.loads(result.pop("data_json"))
    return result


def delete_meeting(meeting_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.commit()
    conn.close()


# ============================================================================
# SECTION 2: NLP ENGINE (transcription, extraction, Q&A) — GEMINI VERSION
# ----------------------------------------------------------------------------
# Gemini has an OpenAI-compatible endpoint, so we keep using the `openai`
# package but point it at Google's base_url with a Gemini API key.
# Docs: https://ai.google.dev/gemini-api/docs/openai
# ============================================================================

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EXTRACTION_MODEL = "gemini-2.5-flash"


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def transcribe_audio(file_path: str, api_key: str) -> str:
    """
    Transcribe audio using Gemini directly (native SDK, not the OpenAI
    compat layer — Gemini's audio understanding isn't exposed through the
    OpenAI-style /audio/transcriptions endpoint the way Whisper is).
    Requires: pip install google-genai
    """
    from google import genai as google_genai

    client = google_genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=file_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            uploaded,
            "Transcribe this audio verbatim. If multiple speakers are "
            "audible, label each line with a speaker name or 'Speaker 1', "
            "'Speaker 2', etc. Return only the transcript text.",
        ],
    )
    return response.text


EXTRACTION_SCHEMA_PROMPT = """You are an assistant that extracts structured information from a meeting transcript.

Read the transcript below and return ONLY a valid JSON object (no markdown fences, no commentary) with this exact shape:

{
  "summary": "2-4 sentence summary of the whole meeting",
  "key_points": ["short discussion point", "..."],
  "decisions": ["decision made", "..."],
  "action_items": [
    {"task": "...", "person": "name or 'Unassigned'", "deadline": "date or 'Not specified'", "priority": "High" | "Medium" | "Low"}
  ],
  "entities": {
    "people": ["..."],
    "organizations": ["..."],
    "dates": ["..."],
    "projects": ["..."],
    "technologies": ["..."]
  }
}

Rules for priority:
- High: urgent, blocking, or explicitly said to be urgent/critical/ASAP, or due very soon
- Medium: normal work item with a deadline
- Low: nice-to-have, no urgency, no near deadline

If the transcript has speaker labels like "Alice: ..." use those names for "person" and "people". If no names are given, use "Unassigned" and infer roles where possible.

Transcript:
---
{transcript}
---

Return ONLY the JSON object.
"""


def extract_meeting_info(transcript: str, api_key: str) -> dict:
    client = _client(api_key)
    prompt = EXTRACTION_SCHEMA_PROMPT.replace("{transcript}", transcript[:15000])

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    # Gemini sometimes wraps JSON in markdown fences even when asked not to; strip defensively.
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    data = json.loads(raw)

    data.setdefault("summary", "")
    data.setdefault("key_points", [])
    data.setdefault("decisions", [])
    data.setdefault("action_items", [])
    data.setdefault("entities", {})
    for key in ("people", "organizations", "dates", "projects", "technologies"):
        data["entities"].setdefault(key, [])

    for item in data["action_items"]:
        item.setdefault("status", "Pending")
        item.setdefault("priority", "Medium")
        item.setdefault("person", "Unassigned")
        item.setdefault("deadline", "Not specified")

    return data


def answer_question(transcript: str, extracted_data: dict, question: str, api_key: str, asking_as: str | None = None) -> str:
    client = _client(api_key)

    context = f"""MEETING TRANSCRIPT:
{transcript[:12000]}

STRUCTURED DATA ALREADY EXTRACTED (summary/decisions/action items):
{json.dumps(extracted_data, indent=2)[:4000]}
"""
    who = f'\nThe person asking is named "{asking_as}". If the question refers to "me"/"my", treat it as referring to this person.' if asking_as else ""

    prompt = f"""You are a meeting assistant. Answer the user's question using ONLY the information in the context below. If the answer isn't in the context, say so honestly — do not make anything up.{who}

{context}

QUESTION: {question}

Answer concisely and directly.
"""
    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ============================================================================
# SECTION 3: PDF REPORT GENERATOR
# ============================================================================

def _safe(text) -> str:
    if text is None:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")


class MeetingPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, _safe(self.title_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _safe(f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(235, 235, 245)
        self.cell(0, 9, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullet_list(self, items):
        self.set_font("Helvetica", "", 11)
        for item in items:
            self.multi_cell(0, 6, _safe(f"- {item}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)


def generate_pdf(meeting_title: str, data: dict) -> bytes:
    pdf = MeetingPDF()
    pdf.title_text = meeting_title
    pdf.add_page()

    pdf.section_title("Meeting Summary")
    pdf.body_text(data.get("summary", "N/A"))

    pdf.section_title("Key Discussion Points")
    pdf.bullet_list(data.get("key_points", []) or ["None recorded"])

    pdf.section_title("Decisions Made")
    pdf.bullet_list(data.get("decisions", []) or ["None recorded"])

    pdf.section_title("Action Items")
    pdf.set_font("Helvetica", "B", 10)
    col_widths = [70, 35, 30, 25, 25]
    headers = ["Task", "Person", "Deadline", "Priority", "Status"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, _safe(h), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for item in data.get("action_items", []):
        pdf.cell(col_widths[0], 8, _safe(item.get("task", ""))[:45], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[1], 8, _safe(item.get("person", ""))[:20], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[2], 8, _safe(item.get("deadline", ""))[:15], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[3], 8, _safe(item.get("priority", ""))[:12], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[4], 8, _safe(item.get("status", ""))[:12], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()
    pdf.ln(4)

    pdf.section_title("Important Entities")
    entities = data.get("entities", {})
    for label, key in [("People", "people"), ("Organizations", "organizations"),
                        ("Dates", "dates"), ("Projects", "projects"), ("Technologies", "technologies")]:
        values = entities.get(key, [])
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, _safe(f"{label}:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _safe(", ".join(values) if values else "None found"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    return bytes(pdf.output())


# ============================================================================
# SECTION 4: TRANSLATION (free, via deep-translator / Google Translate)
# ============================================================================

LANGUAGES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "French": "fr", "German": "de",
    "Spanish": "es", "Chinese (Simplified)": "zh-CN", "Japanese": "ja", "Arabic": "ar",
}
_CHUNK_SIZE = 4500


def translate_text(text: str, target_lang_code: str) -> str:
    if not text or target_lang_code == "en":
        return text
    translator = GoogleTranslator(source="auto", target=target_lang_code)
    chunks = [text[i:i + _CHUNK_SIZE] for i in range(0, len(text), _CHUNK_SIZE)]
    return " ".join(translator.translate(chunk) for chunk in chunks)


def translate_meeting_data(data: dict, target_lang_code: str) -> dict:
    if target_lang_code == "en":
        return data
    translated = dict(data)
    translated["summary"] = translate_text(data.get("summary", ""), target_lang_code)
    translated["key_points"] = [translate_text(p, target_lang_code) for p in data.get("key_points", [])]
    translated["decisions"] = [translate_text(d, target_lang_code) for d in data.get("decisions", [])]
    translated["action_items"] = [
        {**item, "task": translate_text(item.get("task", ""), target_lang_code)}
        for item in data.get("action_items", [])
    ]
    return translated


# ============================================================================
# SECTION 5: EXPORTERS (email drafts, CSV/JSON, Trello)
# ============================================================================

EMAIL_TEMPLATE = """Subject: Action Item: {task}

Hi {person},

Following today's meeting ("{meeting_title}"), you've been assigned the following task:

  Task: {task}
  Deadline: {deadline}
  Priority: {priority}

Please let me know if you have any questions or need help prioritizing this
against your other work.

Thanks,
Meeting-to-Action-Items AI
"""


def generate_email_drafts(meeting_title: str, action_items: list[dict]) -> list[dict]:
    drafts = []
    for item in action_items:
        body = EMAIL_TEMPLATE.format(
            task=item.get("task", ""), person=item.get("person", "Team"),
            deadline=item.get("deadline", "Not specified"), priority=item.get("priority", "Medium"),
            meeting_title=meeting_title,
        )
        drafts.append({"person": item.get("person", "Unassigned"),
                        "subject": f"Action Item: {item.get('task','')}", "body": body})
    return drafts


def action_items_to_csv(action_items: list[dict]) -> bytes:
    output = io.StringIO()
    fieldnames = ["task", "person", "deadline", "priority", "status"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for item in action_items:
        writer.writerow({k: item.get(k, "") for k in fieldnames})
    return output.getvalue().encode("utf-8")


def action_items_to_json(action_items: list[dict]) -> bytes:
    return json.dumps(action_items, indent=2).encode("utf-8")


def export_to_trello(action_items: list[dict], api_key: str, token: str, list_id: str) -> list[dict]:
    results = []
    url = "https://api.trello.com/1/cards"
    for item in action_items:
        params = {
            "key": api_key, "token": token, "idList": list_id,
            "name": item.get("task", "Untitled task"),
            "desc": f"Assigned to: {item.get('person','Unassigned')}\nPriority: {item.get('priority','Medium')}",
        }
        deadline = item.get("deadline")
        if deadline and deadline.lower() != "not specified":
            params["due"] = deadline
        try:
            resp = requests.post(url, params=params, timeout=10)
            if resp.status_code in (200, 201):
                results.append({"task": item.get("task"), "status": "success"})
            else:
                results.append({"task": item.get("task"), "status": "failed", "error": resp.text[:200]})
        except Exception as e:
            results.append({"task": item.get("task"), "status": "failed", "error": str(e)})
    return results


# ============================================================================
# SECTION 6: STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Meeting-to-Action-Items AI", page_icon="🗒️", layout="wide")
init_db()

if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = None

# ----------------------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --brand: #6366F1;
        --brand-dark: #4338CA;
        --bg-soft: #F8F9FC;
        --border-soft: #E5E7EB;
    }

    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* App background */
    .stApp { background-color: var(--bg-soft); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #F3F4F8 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--brand-dark);
        padding-bottom: 0.2rem;
    }

    /* Headings */
    h1, h2, h3 { color: #1F2430; font-weight: 750; }
    h1 { letter-spacing: -0.5px; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--brand);
        background-color: var(--brand);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background-color: var(--brand-dark);
        border-color: var(--brand-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.25);
    }
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid var(--brand);
        color: var(--brand-dark);
        font-weight: 600;
        background-color: white;
    }
    .stDownloadButton > button:hover {
        background-color: #EEF0FF;
    }

    /* Cards: tabs, expanders, containers */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-baseweb="tab-list"] { gap: 4px; }
    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #6B7280;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--brand-dark);
        border-bottom: 3px solid var(--brand);
    }

    /* Metrics / dataframes */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
    }

    /* Text area / inputs */
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid var(--border-soft) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--brand) !important;
        box-shadow: 0 0 0 1px var(--brand) !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] { border-radius: 10px; }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🗒️ Meeting AI")
    st.caption("NLP-powered meeting summarizer, action-item extractor & assistant")

    # Read the Gemini key from Streamlit secrets only. No key input is shown
    # to the user — the app author (you) supplies it once via
    # Manage app > Settings > Secrets as GEMINI_API_KEY = "...".
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    st.divider()
    page = st.radio("Navigate", ["🎙️ New Meeting", "📚 Meeting History"])

    st.divider()
    st.caption("Report language")
    lang_name = st.selectbox("Translate report into", list(LANGUAGES.keys()), index=0)
    lang_code = LANGUAGES[lang_name]

if not api_key:
    st.error(
        "⚠️ No Gemini API key configured for this app. The app owner needs to "
        "set **GEMINI_API_KEY** under *Manage app → Settings → Secrets* in "
        "Streamlit Cloud, then reboot the app."
    )
    st.stop()


def render_dashboard(meeting_id: int, title: str, transcript: str, data: dict):
    tabs = st.tabs(["📋 Overview", "✅ Action Items", "📊 Charts & Entities",
                     "🔍 Search", "💬 Ask Questions", "📤 Export"])

    with tabs[0]:
        st.subheader("Meeting Summary")
        st.write(data.get("summary", "N/A"))
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Key Discussion Points")
            for point in data.get("key_points", []):
                st.markdown(f"- {point}")
        with col2:
            st.subheader("🎯 Decisions Made")
            for dec in data.get("decisions", []):
                st.markdown(f"- {dec}")

    with tabs[1]:
        st.subheader("✅ Action Items")
        items = data.get("action_items", [])
        if items:
            df = pd.DataFrame(items)
            edited = st.data_editor(
                df,
                column_config={
                    "priority": st.column_config.SelectboxColumn(options=["High", "Medium", "Low"]),
                    "status": st.column_config.SelectboxColumn(options=["Pending", "In Progress", "Completed"]),
                },
                num_rows="dynamic", use_container_width=True, key=f"editor_{meeting_id}",
            )
            if st.button("💾 Save changes", key=f"save_{meeting_id}"):
                data["action_items"] = edited.to_dict(orient="records")
                update_meeting_data(meeting_id, data)
                st.success("Saved.")
                st.rerun()
        else:
            st.info("No action items were extracted from this meeting.")

    with tabs[2]:
        items = data.get("action_items", [])
        if items:
            df = pd.DataFrame(items)
            c1, c2 = st.columns(2)
            with c1:
                by_person = df.groupby("person").size().reset_index(name="count")
                st.plotly_chart(px.bar(by_person, x="person", y="count", title="📊 Action Items by Person"),
                                 use_container_width=True)
            with c2:
                by_status = df.groupby("status").size().reset_index(name="count")
                st.plotly_chart(px.pie(by_status, names="status", values="count", title="📈 Task Status"),
                                 use_container_width=True)
            by_priority = df.groupby("priority").size().reset_index(name="count")
            fig3 = px.bar(by_priority, x="priority", y="count", title="🚦 Priority Breakdown",
                          category_orders={"priority": ["High", "Medium", "Low"]}, color="priority",
                          color_discrete_map={"High": "#e74c3c", "Medium": "#f1c40f", "Low": "#2ecc71"})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No action items to chart yet.")

        st.subheader("🏷️ Important Entities")
        entities = data.get("entities", {})
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown("**People**"); st.write(", ".join(entities.get("people", [])) or "—")
            st.markdown("**Organizations**"); st.write(", ".join(entities.get("organizations", [])) or "—")
        with ec2:
            st.markdown("**Dates**"); st.write(", ".join(entities.get("dates", [])) or "—")
            st.markdown("**Projects**"); st.write(", ".join(entities.get("projects", [])) or "—")
        with ec3:
            st.markdown("**Technologies**"); st.write(", ".join(entities.get("technologies", [])) or "—")

    with tabs[3]:
        st.subheader("🔍 Search within transcript")
        query = st.text_input("Search term", key=f"search_{meeting_id}")
        if query:
            lines = re.split(r"(?<=[.!?])\s+|\n", transcript)
            matches = [l for l in lines if query.lower() in l.lower()]
            st.caption(f"{len(matches)} match(es)")
            for m in matches:
                st.markdown(f"> {re.sub(f'(?i)({re.escape(query)})', r'**\\1**', m)}")
        with st.expander("View full transcript"):
            st.text_area("Transcript", transcript, height=300, key=f"full_transcript_{meeting_id}")

    with tabs[4]:
        st.subheader("💬 Ask questions about this meeting")
        your_name = st.text_input("Your name (so 'what are MY tasks' works)", key=f"name_{meeting_id}")
        question = st.text_input("Your question", placeholder="What tasks were assigned to me?", key=f"q_{meeting_id}")
        if st.button("Ask", key=f"ask_{meeting_id}"):
            if not question:
                st.warning("Type a question first.")
            else:
                with st.spinner("Thinking..."):
                    answer = answer_question(transcript, data, question, api_key, asking_as=your_name or None)
                st.markdown(f"**Answer:** {answer}")

    with tabs[5]:
        st.subheader("📤 Export & share")

        st.markdown(f"**Translate report to {lang_name}**")
        if st.button("🌐 Translate report", key=f"translate_{meeting_id}"):
            with st.spinner(f"Translating into {lang_name}..."):
                st.session_state[f"translated_{meeting_id}"] = translate_meeting_data(data, lang_code)
            st.success("Translated below (does not overwrite your saved English data).")

        translated_data = st.session_state.get(f"translated_{meeting_id}")
        export_data = translated_data if translated_data else data
        if translated_data:
            st.info(f"Showing/exporting the {lang_name} version.")
            st.write(export_data.get("summary", ""))

        st.divider()
        pdf_bytes = generate_pdf(title, export_data)
        st.download_button("⬇️ Download PDF Report", data=pdf_bytes,
                            file_name=f"{title.replace(' ', '_')}_report.pdf", mime="application/pdf")

        items = export_data.get("action_items", [])
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Export tasks as CSV", data=action_items_to_csv(items),
                                file_name="action_items.csv", mime="text/csv")
        with c2:
            st.download_button("⬇️ Export tasks as JSON", data=action_items_to_json(items),
                                file_name="action_items.json", mime="application/json")

        st.divider()
        st.markdown("**✉️ Email drafts (one per action item)**")
        for d in generate_email_drafts(title, items):
            with st.expander(f"To: {d['person']} — {d['subject']}"):
                st.code(d["body"])

        st.divider()
        st.markdown("**🔗 Export tasks to Trello**")
        with st.expander("Trello integration"):
            st.caption("Get your key/token from https://trello.com/power-ups/admin, and the target list's ID from the Trello API.")
            t_key = st.text_input("Trello API Key", key=f"trello_key_{meeting_id}")
            t_token = st.text_input("Trello Token", type="password", key=f"trello_token_{meeting_id}")
            t_list = st.text_input("Trello List ID", key=f"trello_list_{meeting_id}")
            if st.button("Push tasks to Trello", key=f"trello_btn_{meeting_id}"):
                if not (t_key and t_token and t_list):
                    st.warning("Fill in all three Trello fields first.")
                else:
                    with st.spinner("Pushing cards to Trello..."):
                        results = export_to_trello(items, t_key, t_token, t_list)
                    for r in results:
                        if r["status"] == "success":
                            st.success(f"✅ {r['task']}")
                        else:
                            st.error(f"❌ {r['task']}: {r.get('error')}")


# ---- Page: New Meeting ----
if page == "🎙️ New Meeting":
    st.header("🎙️ New Meeting")
    meeting_title = st.text_input("Meeting title", value="Untitled Meeting")
    input_mode = st.radio("Input type", ["📄 Transcript text", "🎤 Audio file"], horizontal=True)

    transcript_text = None

    if input_mode == "📄 Transcript text":
        st.caption("Tip: if your transcript has speaker labels like `Alice: ...` on each line, "
                    "the AI will use those names for action items.")
        upload = st.file_uploader("Upload a .txt or .docx transcript (optional)", type=["txt", "docx"])
        pasted = st.text_area("...or paste the transcript here", height=200)

        if upload is not None:
            if upload.name.endswith(".docx"):
                from docx import Document
                doc = Document(upload)
                transcript_text = "\n".join(p.text for p in doc.paragraphs)
            else:
                transcript_text = upload.read().decode("utf-8", errors="ignore")
        elif pasted.strip():
            transcript_text = pasted

    else:
        audio_file = st.file_uploader("Upload meeting audio", type=["mp3", "wav", "m4a", "mp4"])
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("🎧 Transcribe audio"):
                with st.spinner("Transcribing... (this can take a minute for longer recordings)"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    transcript_text = transcribe_audio(tmp_path, api_key)
                st.session_state["pending_transcript"] = transcript_text
                st.success("Transcription complete — review below, then click Process.")

        transcript_text = st.session_state.get("pending_transcript")
        if transcript_text:
            transcript_text = st.text_area("Transcribed text (edit if needed)", transcript_text, height=200)

    st.divider()
    if st.button("🧠 Process Meeting", type="primary"):
        if not transcript_text or not transcript_text.strip():
            st.warning("Provide a transcript (paste, upload, or transcribe audio) first.")
        else:
            with st.spinner("Running NLP pipeline: summarizing, extracting action items, detecting priority..."):
                data = extract_meeting_info(transcript_text, api_key)
                meeting_id = save_meeting(meeting_title, transcript_text, data)
            st.session_state.current_meeting_id = meeting_id
            st.session_state.pop("pending_transcript", None)
            st.success("Done! Dashboard below.")
            st.rerun()

    if st.session_state.current_meeting_id:
        meeting = get_meeting(st.session_state.current_meeting_id)
        if meeting:
            st.divider()
            st.header(f"📊 Dashboard — {meeting['title']}")
            render_dashboard(meeting["id"], meeting["title"], meeting["transcript"], meeting["data"])

# ---- Page: Meeting History ----
else:
    st.header("📚 Meeting History")
    meetings = get_all_meetings()
    if not meetings:
        st.info("No meetings processed yet. Go to 'New Meeting' to get started.")
    else:
        labels = [f"#{m['id']} — {m['title']} ({m['created_at']})" for m in meetings]
        selected = st.selectbox("Select a past meeting", labels)
        selected_id = meetings[labels.index(selected)]["id"]

        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🗑️ Delete this meeting"):
                delete_meeting(selected_id)
                st.rerun()

        meeting = get_meeting(selected_id)
        if meeting:
            render_dashboard(meeting["id"], meeting["title"], meeting["transcript"], meeting["data"])
