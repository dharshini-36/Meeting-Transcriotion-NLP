import re
from datetime import datetime
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Meeting Intelligence AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Meeting Intelligence AI")
st.caption("Convert meeting audio or transcripts into summaries, decisions, action items and deadlines.")

# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_classifier():
    # BART-large-MNLI is used as a zero-shot text classifier.
    # It is an NLP transformer model and works without a custom training dataset.
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

classifier = load_classifier()

LABELS = [
    "action item",
    "decision",
    "discussion",
    "question",
    "information or update"
]

# -----------------------------
# Helper functions
# -----------------------------
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def classify_sentences(sentences):
    rows = []

    for sentence in sentences:
        result = classifier(
            sentence,
            candidate_labels=LABELS,
            multi_label=False
        )

        rows.append({
            "Sentence": sentence,
            "Category": result["labels"][0],
            "Confidence": round(float(result["scores"][0]) * 100, 2)
        })

    return pd.DataFrame(rows)


def extract_people(text):
    # Detect simple "Name:" speaker patterns such as Rahul: or Priya:
    names = re.findall(r"\b([A-Z][a-z]{2,20})\s*:", text)
    return sorted(set(names))


def extract_dates(text):
    patterns = [
        r"\b(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(?:next week|next month|this week|this month)\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:,\s*\d{4})?\b"
    ]

    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    return sorted(set(found), key=str.lower)


def extract_entities(text):
    people = extract_people(text)

    # Simple project/organization candidates from capitalized multi-word names.
    organizations = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9&.-]*\s+){1,3}(?:Technologies|Technology|Solutions|Company|Corporation|University|College|Inc|Ltd)\b",
        text
    )

    return {
        "People": sorted(set(people)),
        "Organizations": sorted(set(organizations))
    }


def extract_action_items(classified_df):
    actions = classified_df[
        classified_df["Category"].str.lower() == "action item"
    ].copy()

    result = []

    for _, row in actions.iterrows():
        sentence = row["Sentence"]

        # Try to identify an explicit speaker.
        person_match = re.match(r"^\s*([A-Z][a-z]{2,20})\s*:", sentence)
        person = person_match.group(1) if person_match else "Not specified"

        # Remove speaker name from task text.
        task = re.sub(r"^\s*[A-Z][a-z]{2,20}\s*:\s*", "", sentence)

        result.append({
            "Assigned Person": person,
            "Action Item": task,
            "Deadline": "Not specified",
            "Confidence": row["Confidence"]
        })

    return pd.DataFrame(result)


def build_summary(classified_df):
    if classified_df.empty:
        return "No meeting content was detected."

    important = classified_df[
        classified_df["Category"].str.lower().isin(
            ["action item", "decision", "discussion", "information or update"]
        )
    ]

    if important.empty:
        return "The meeting did not contain enough classified information."

    sentences = important["Sentence"].tolist()[:5]
    return " ".join(sentences)


def create_report(summary, discussions, decisions, actions, entities, dates):
    lines = [
        "# Meeting Intelligence Report",
        "",
        "## Meeting Summary",
        summary,
        "",
        "## Key Discussion Points"
    ]

    for item in discussions:
        lines.append(f"- {item}")

    lines += ["", "## Decisions Made"]

    for item in decisions:
        lines.append(f"- {item}")

    lines += ["", "## Action Items"]

    if actions.empty:
        lines.append("- No action items detected.")
    else:
        for _, row in actions.iterrows():
            lines.append(
                f"- {row['Assigned Person']}: {row['Action Item']} "
                f"(Deadline: {row['Deadline']})"
            )

    lines += ["", "## Important Entities"]
    lines.append(f"- People: {', '.join(entities['People']) or 'None detected'}")
    lines.append(
        f"- Organizations: {', '.join(entities['Organizations']) or 'None detected'}"
    )
    lines.append(f"- Dates/Deadlines: {', '.join(dates) or 'None detected'}")

    return "\n".join(lines)


# -----------------------------
# Input section
# -----------------------------
st.sidebar.header("Input")

input_type = st.sidebar.radio(
    "Choose input type",
    ["Transcript", "Audio"]
)

transcript = ""

if input_type == "Transcript":
    transcript = st.text_area(
        "Paste meeting transcript",
        height=300,
        placeholder=(
            "Example:\n"
            "Rahul: We need to finish the homepage by Friday.\n"
            "Dharshini: I'll handle the frontend implementation.\n"
            "Priya: I'll prepare the test cases tomorrow.\n"
            "Rahul: Let's review everything on Friday."
        )
    )

else:
    audio_file = st.file_uploader(
        "Upload meeting audio",
        type=["wav", "mp3", "m4a", "ogg"]
    )

    st.info(
        "Audio transcription is optional in this starter version. "
        "To enable automatic speech-to-text, add a Whisper model/API and "
        "pass its transcript to the NLP pipeline."
    )

    if audio_file:
        st.audio(audio_file)

        st.warning(
            "For the current version, paste the generated transcript below "
            "after transcribing the audio."
        )

        transcript = st.text_area(
            "Paste audio transcript",
            height=250
        )


if st.button("🚀 Analyze Meeting", type="primary"):
    if not transcript.strip():
        st.error("Please provide a meeting transcript.")
        st.stop()

    with st.spinner("Analyzing meeting..."):
        sentences = split_sentences(transcript)
        classified_df = classify_sentences(sentences)

        actions = extract_action_items(classified_df)

        discussions = classified_df[
            classified_df["Category"].str.lower() == "discussion"
        ]["Sentence"].tolist()

        decisions = classified_df[
            classified_df["Category"].str.lower() == "decision"
        ]["Sentence"].tolist()

        summary = build_summary(classified_df)
        entities = extract_entities(transcript)
        dates = extract_dates(transcript)

    st.success("Meeting analysis completed.")

    # -----------------------------
    # Dashboard metrics
    # -----------------------------
    st.subheader("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sentences", len(classified_df))

    with col2:
        st.metric("Action Items", len(actions))

    with col3:
        st.metric("Decisions", len(decisions))

    with col4:
        st.metric("Deadlines Found", len(dates))

    # -----------------------------
    # Summary
    # -----------------------------
    st.subheader("📋 Meeting Summary")
    st.write(summary)

    # -----------------------------
    # Visual category chart
    # -----------------------------
    st.subheader("📈 Conversation Analysis")

    category_counts = classified_df["Category"].value_counts()
    st.bar_chart(category_counts)

    # -----------------------------
    # Discussions
    # -----------------------------
    st.subheader("💬 Key Discussion Points")

    if discussions:
        for item in discussions:
            st.write(f"• {item}")
    else:
        st.write("No discussion points detected.")

    # -----------------------------
    # Decisions
    # -----------------------------
    st.subheader("🎯 Decisions Made")

    if decisions:
        for item in decisions:
            st.write(f"• {item}")
    else:
        st.write("No decisions detected.")

    # -----------------------------
    # Action items
    # -----------------------------
    st.subheader("✅ Action Items")

    if actions.empty:
        st.write("No action items detected.")
    else:
        st.dataframe(
            actions,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------
    # Entities
    # -----------------------------
    st.subheader("🏷️ Important Entities")

    entity_col1, entity_col2 = st.columns(2)

    with entity_col1:
        st.write("**People**")
        for person in entities["People"]:
            st.write(f"• {person}")

    with entity_col2:
        st.write("**Organizations**")
        for organization in entities["Organizations"]:
            st.write(f"• {organization}")

    st.write("**Dates / Deadlines**")
    if dates:
        for date in dates:
            st.write(f"• {date}")
    else:
        st.write("No dates detected.")

    # -----------------------------
    # Classification table
    # -----------------------------
    with st.expander("🔍 View NLP Classification Details"):
        st.dataframe(
            classified_df,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------
    # Downloadable report
    # -----------------------------
    report = create_report(
        summary,
        discussions,
        decisions,
        actions,
        entities,
        dates
    )

    st.subheader("📄 Meeting Report")

    st.download_button(
        label="⬇️ Download Meeting Report",
        data=report,
        file_name="meeting_report.md",
        mime="text/markdown"
    )

    st.caption(
        "The dashboard provides visual and text-based results, while the "
        "downloadable report provides a structured text report."
    )
