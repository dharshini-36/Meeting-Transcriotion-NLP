import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from src.nlp_pipeline import (
    clean_transcript, extract_entities, summarize_text,
    extract_key_points, extract_decisions, extract_action_items,
)
from src.priority import annotate_priorities
from src.speaker import parse_speakers, build_speaker_char_map, plain_transcript
from src.qa_engine import answer_question, search_transcript
from src.translator import translate_text, LANGUAGES
from src.database import save_meeting, list_meetings, get_meeting, delete_meeting
from src.export_utils import (
    action_items_to_csv_bytes, action_items_to_ics_bytes,
    report_to_docx_bytes, report_to_pdf_bytes,
)
from src.email_generator import generate_all_drafts

st.set_page_config(page_title="Meeting-to-Action-Items AI", page_icon="🎤", layout="wide")

# ---------------------------------------------------------------- session state
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎤 Meeting AI")
    st.caption("NLP-powered meeting summarizer & task extractor")
    st.session_state.user_name = st.text_input(
        "Your name (for 'my tasks' Q&A)", value=st.session_state.user_name
    )
    page = st.radio(
        "Navigate",
        ["🆕 New Meeting", "📊 Dashboard", "❓ Ask Questions", "🔎 Search",
         "📜 History", "📤 Export"],
    )
    st.divider()
    st.caption("Built with spaCy NER + DistilBART summarization + "
               "zero-shot DistilBERT priority classification.")


def run_pipeline(transcript_raw: str, meeting_title: str):
    transcript_raw = clean_transcript(transcript_raw)
    segments, speaker_counts = parse_speakers(transcript_raw)
    char_map = build_speaker_char_map(segments)
    plain_text = plain_transcript(segments) if segments else transcript_raw

    with st.spinner("Summarizing meeting..."):
        summary = summarize_text(plain_text)
    with st.spinner("Extracting entities..."):
        entities = extract_entities(plain_text)
    with st.spinner("Identifying key discussion points..."):
        key_points = extract_key_points(plain_text)
    with st.spinner("Detecting decisions..."):
        decisions = extract_decisions(plain_text)
    with st.spinner("Extracting action items..."):
        action_items = extract_action_items(plain_text, speaker_map=char_map)
    with st.spinner("Scoring priority..."):
        action_items = annotate_priorities(action_items)
    for item in action_items:
        item["status"] = "Pending"

    data = {
        "title": meeting_title,
        "transcript": transcript_raw,
        "plain_text": plain_text,
        "summary": summary,
        "entities": entities,
        "key_points": key_points,
        "decisions": decisions,
        "action_items": action_items,
        "speaker_counts": speaker_counts,
    }
    st.session_state.analysis = data
    save_meeting(meeting_title, transcript_raw, summary, data)
    return data


# ---------------------------------------------------------------- New Meeting
if page == "🆕 New Meeting":
    st.header("New Meeting")
    meeting_title = st.text_input("Meeting title", value=f"Meeting {datetime.date.today()}")

    input_mode = st.radio("Input type", ["📄 Paste / upload transcript", "🎧 Upload audio"], horizontal=True)

    transcript_text = ""
    if input_mode == "📄 Paste / upload transcript":
        uploaded = st.file_uploader("Upload a .txt transcript (optional)", type=["txt"])
        if uploaded:
            transcript_text = uploaded.read().decode("utf-8", errors="ignore")
        transcript_text = st.text_area(
            "Or paste transcript here (tip: prefix lines with 'Name: ' for speaker identification)",
            value=transcript_text, height=280,
            placeholder="Dharshini: I'll fix the login bug by Friday.\nPriya: I'll prepare test cases by Thursday.\n...",
        )
    else:
        audio_file = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
        st.caption("Uses a free online speech-to-text service - best for short, clear recordings.")
        if audio_file and st.button("Transcribe audio"):
            from src.speech_to_text import audio_file_to_wav, transcribe_wav
            with st.spinner("Converting and transcribing audio... this can take a while."):
                wav_path = audio_file_to_wav(audio_file)
                transcript_text = transcribe_wav(wav_path)
            st.success("Transcription complete - review/edit below before analyzing.")
        transcript_text = st.text_area("Transcript (from audio)", value=transcript_text, height=280)

    if st.button("🧠 Analyze Meeting", type="primary", disabled=not transcript_text.strip()):
        run_pipeline(transcript_text, meeting_title)
        st.success("Analysis complete! Head to the Dashboard tab.")

# ---------------------------------------------------------------- Dashboard
elif page == "📊 Dashboard":
    data = st.session_state.analysis
    if not data:
        st.info("No meeting analyzed yet. Go to **New Meeting** first, or load one from **History**.")
    else:
        st.header(f"📊 Dashboard — {data['title']}")

        st.subheader("📋 Meeting Summary")
        st.write(data["summary"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Key Discussion Points")
            for p in data["key_points"]:
                st.markdown(f"- {p}")
        with col2:
            st.subheader("🎯 Decisions Made")
            if data["decisions"]:
                for d in data["decisions"]:
                    st.markdown(f"- {d}")
            else:
                st.caption("No explicit decisions detected.")

        st.subheader("✅ Action Items")
        df = pd.DataFrame(data["action_items"])
        if not df.empty:
            edited = st.data_editor(
                df[["person", "task", "deadline", "priority", "status"]],
                column_config={
                    "status": st.column_config.SelectboxColumn(
                        options=["Pending", "In Progress", "Completed"]
                    ),
                    "priority": st.column_config.SelectboxColumn(
                        options=["High", "Medium", "Low"]
                    ),
                },
                num_rows="dynamic", use_container_width=True, key="editor",
            )
            # persist edits back into session state
            data["action_items"] = edited.to_dict("records")
        else:
            st.caption("No action items detected.")

        st.subheader("🏷️ Important Entities")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("People", len(data["entities"]["people"]))
        e2.metric("Organizations", len(data["entities"]["organizations"]))
        e3.metric("Dates", len(data["entities"]["dates"]))
        e4.metric("Projects/Tech", len(data["entities"]["projects_tech"]))
        with st.expander("View entity details"):
            st.json(data["entities"])

        st.subheader("📈 Visual Insights")
        v1, v2 = st.columns(2)
        with v1:
            if data["action_items"]:
                counts = pd.Series([i["person"] for i in data["action_items"]]).value_counts()
                fig, ax = plt.subplots()
                counts.plot(kind="barh", ax=ax, color="#4F46E5")
                ax.set_xlabel("Number of tasks")
                ax.set_title("Action Items by Person")
                st.pyplot(fig)
        with v2:
            if data["action_items"]:
                status_counts = pd.Series([i.get("status", "Pending") for i in data["action_items"]]).value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(status_counts, labels=status_counts.index, autopct="%1.0f%%",
                        colors=["#F59E0B", "#3B82F6", "#10B981"])
                ax2.set_title("Task Status")
                st.pyplot(fig2)

        if data["speaker_counts"] and len(data["speaker_counts"]) > 1:
            st.subheader("🎙️ Speaker Participation")
            sp_df = pd.Series(data["speaker_counts"]).sort_values(ascending=False)
            st.bar_chart(sp_df)

        st.subheader("📅 Upcoming Deadlines")
        deadline_df = pd.DataFrame(
            [i for i in data["action_items"] if i["deadline"] != "Not specified"]
        )
        if not deadline_df.empty:
            st.dataframe(deadline_df[["person", "task", "deadline"]], use_container_width=True)
        else:
            st.caption("No explicit deadlines detected.")

        st.subheader("📧 Generated Email Drafts")
        drafts = generate_all_drafts(data["action_items"], data["title"])
        if drafts:
            for d in drafts:
                with st.expander(f"✉️ To: {d['to']} — {d['subject']}"):
                    st.text(d["body"])
        else:
            st.caption("No assigned-person action items to draft emails for yet.")

# ---------------------------------------------------------------- Ask Questions
elif page == "❓ Ask Questions":
    data = st.session_state.analysis
    if not data:
        st.info("Analyze a meeting first.")
    else:
        st.header("❓ Ask About This Meeting")
        st.caption('Try: "What tasks were assigned to me?" or "What did we decide about the homepage?"')
        q = st.text_input("Your question")
        if st.button("Ask", type="primary", disabled=not q.strip()):
            with st.spinner("Thinking..."):
                answer = answer_question(q, data["plain_text"], data["action_items"], st.session_state.user_name)
            st.markdown(f"**Answer:** {answer}")

# ---------------------------------------------------------------- Search
elif page == "🔎 Search":
    data = st.session_state.analysis
    if not data:
        st.info("Analyze a meeting first.")
    else:
        st.header("🔎 Search Transcript")
        query = st.text_input("Search term")
        if query:
            results = search_transcript(data["transcript"], query)
            st.caption(f"{len(results)} match(es) found")
            for r in results:
                st.markdown(f"> {r}")

# ---------------------------------------------------------------- History
elif page == "📜 History":
    st.header("📜 Meeting History")
    meetings = list_meetings()
    if not meetings:
        st.info("No meetings saved yet.")
    else:
        for m in meetings:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.markdown(f"**{m['title']}**  \n_{m['created_at']}_  \n{m['summary'][:150]}...")
                if c2.button("Open", key=f"open_{m['id']}"):
                    full = get_meeting(m["id"])
                    st.session_state.analysis = full["data"]
                    st.success(f"Loaded '{m['title']}'. Go to Dashboard.")
                if c3.button("Delete", key=f"del_{m['id']}"):
                    delete_meeting(m["id"])
                    st.rerun()

# ---------------------------------------------------------------- Export
elif page == "📤 Export":
    data = st.session_state.analysis
    if not data:
        st.info("Analyze a meeting first.")
    else:
        st.header("📤 Export & Translate")

        st.subheader("Downloadable Report")
        c1, c2 = st.columns(2)
        with c1:
            docx_bytes = report_to_docx_bytes(
                data["title"], data["summary"], data["key_points"],
                data["decisions"], data["action_items"], data["entities"],
            )
            st.download_button("⬇️ Download Report (DOCX)", docx_bytes,
                                file_name=f"{data['title']}_report.docx")
        with c2:
            pdf_bytes = report_to_pdf_bytes(
                data["title"], data["summary"], data["key_points"],
                data["decisions"], data["action_items"], data["entities"],
            )
            st.download_button("⬇️ Download Report (PDF)", pdf_bytes,
                                file_name=f"{data['title']}_report.pdf")

        st.subheader("Export Tasks to Task-Management Systems")
        st.caption("CSV imports directly into Trello, Asana, Jira, ClickUp and Notion. "
                    ".ics adds deadlines to Google/Outlook/Apple Calendar.")
        c3, c4 = st.columns(2)
        with c3:
            csv_bytes = action_items_to_csv_bytes(data["action_items"])
            st.download_button("⬇️ Tasks as CSV", csv_bytes, file_name=f"{data['title']}_tasks.csv")
        with c4:
            ics_bytes = action_items_to_ics_bytes(data["action_items"], data["title"])
            st.download_button("⬇️ Deadlines as Calendar (.ics)", ics_bytes,
                                file_name=f"{data['title']}_deadlines.ics")

        st.subheader("Translate Report")
        target_lang = st.selectbox("Translate summary + action items to:", list(LANGUAGES.keys()))
        if st.button("Translate"):
            with st.spinner("Translating..."):
                t_summary = translate_text(data["summary"], LANGUAGES[target_lang])
                t_points = [translate_text(p, LANGUAGES[target_lang]) for p in data["key_points"]]
            st.markdown(f"**Summary ({target_lang}):** {t_summary}")
            st.markdown(f"**Key Points ({target_lang}):**")
            for p in t_points:
                st.markdown(f"- {p}")
