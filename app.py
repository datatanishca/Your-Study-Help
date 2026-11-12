import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = "embeddinggemma:300m"
CHAT_MODEL = "qwen3:4b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_RELEVANCE_SCORE = 0.30
SUMMARY_BATCH_CHARS = 30000
PROGRESS_FILE = Path(__file__).with_name("study_progress.json")


def read_pdf_pages(pdf_docs):
    """Extract searchable text while keeping its PDF filename and page number."""
    pages = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"source": pdf.name, "page": page_number, "text": text.strip()})
    return pages


def get_text_chunks(pages):
    """Split each page separately so every passage retains its source location."""
    chunks = []
    for page in pages:
        text = " ".join(page["text"].split())
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            passage = text[start:end].strip()
            if passage:
                chunks.append({"source": page["source"], "page": page["page"], "text": passage})
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def ollama_post(endpoint, payload):
    response = requests.post(f"{OLLAMA_BASE_URL}{endpoint}", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def ollama_chat(prompt, max_tokens=800, json_mode=False):
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    result = ollama_post("/api/chat", payload)
    message = result.get("message", {})
    content = message.get("content", "").strip()
    if content:
        return content
    if message.get("thinking"):
        return "The model did not return a final answer. Please try the question again."
    return ""


def embed_texts(texts):
    embeddings = []
    for start in range(0, len(texts), 32):
        result = ollama_post(
            "/api/embed",
            {"model": EMBEDDING_MODEL, "input": texts[start:start + 32]},
        )
        embeddings.extend(result["embeddings"])
    return embeddings


def get_vectorstore(chunks):
    if not chunks:
        raise ValueError("No selectable text was found in the uploaded PDFs.")
    vectors = np.array(embed_texts([item["text"] for item in chunks]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    return {"chunks": chunks, "vectors": vectors}


def find_evidence(question, vectorstore, limit=5):
    query_vector = np.array(embed_texts([question])[0], dtype=np.float32)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    scores = vectorstore["vectors"] @ query_vector
    indices = np.argsort(scores)[::-1]
    return [
        {**vectorstore["chunks"][int(index)], "score": float(scores[index]), "source_id": f"S{position}"}
        for position, index in enumerate(indices[:limit], start=1)
    ]


def answer_question(question, vectorstore, history):
    evidence = find_evidence(question, vectorstore)
    if not evidence or evidence[0]["score"] < MIN_RELEVANCE_SCORE:
        return (
            "I couldn’t find passages in the uploaded PDFs that clearly support an answer. "
            "Try asking in a different way or check that the relevant PDF was processed.",
            [],
        )
    evidence_text = "\n\n".join(
        f"[{item['source_id']}] File: {item['source']} | Page {item['page']}\nPassage: {item['text']}"
        for item in evidence
    )
    recent_history = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in history[-6:]
    )
    prompt = (
        "Answer the user's question directly in plain, everyday language. Be concise "
        "and give a complete explanation, usually 2–5 sentences or a few short bullets. "
        "For a question asking what a term means, define the term clearly and explain "
        "it simply. The passages below are background for your answer: do not quote "
        "them, mention evidence labels, or describe the retrieval process. If the "
        "passages do not contain enough information, say so clearly. Do not guess or "
        "add unsupported facts.\n\n"
        f"Evidence passages:\n{evidence_text}\n\n"
        f"Recent conversation (for context only):\n{recent_history or '(none)'}\n\n"
        f"Question: {question}"
    )
    answer = ollama_chat(prompt, max_tokens=350)
    return answer or "Ollama returned an empty answer. Please try again.", evidence


def summarize_documents(pages, pdf_names, progress_callback=None):
    """Summarize each uploaded PDF and report progress by PDF, not text batch."""
    pages_by_pdf = {name: [] for name in pdf_names}
    for page in pages:
        pages_by_pdf.setdefault(page["source"], []).append(page)

    summaries = []
    total_pdfs = len(pdf_names)
    for pdf_index, pdf_name in enumerate(pdf_names, start=1):
        if progress_callback:
            progress_callback(pdf_index - 1, total_pdfs, pdf_name, False)
        page_batches = []
        current_batch = []
        current_length = 0
        for page in pages_by_pdf.get(pdf_name, []):
            prefix = f"[{page['source']} | page {page['page']}]\n"
            max_text = max(1000, SUMMARY_BATCH_CHARS - len(prefix))
            for start in range(0, len(page["text"]), max_text):
                passage = prefix + page["text"][start:start + max_text]
                if current_batch and current_length + len(passage) > SUMMARY_BATCH_CHARS:
                    page_batches.append(current_batch)
                    current_batch, current_length = [], 0
                current_batch.append(passage)
                current_length += len(passage)
        if current_batch:
            page_batches.append(current_batch)

        pdf_notes = []
        if not page_batches:
            pdf_notes.append("No selectable text was found in this PDF, so it was skipped.")
        for batch in page_batches:
            note = ollama_chat(
                "Create concise, student-friendly study notes from these PDF pages. "
                "Use simple everyday language. Start with a short overview, list the "
                "key ideas, explain important terms simply, and finish with a few "
                "things to remember. Avoid copying long passages. Keep page numbers "
                "beside the facts they support. Do not add facts not in the pages.\n\n"
                + "\n\n---\n\n".join(batch),
                max_tokens=1000,
            )
            if note.strip():
                pdf_notes.append(note.strip())
        summaries.append(f"## {pdf_name}\n\n" + "\n\n".join(pdf_notes))
        if progress_callback:
            progress_callback(pdf_index, total_pdfs, pdf_name, True)
    return "\n\n".join(summaries) or "I couldn't generate a study summary."


def sampled_chunks(vectorstore, limit=12):
    chunks = vectorstore["chunks"]
    if len(chunks) <= limit:
        indices = list(range(len(chunks)))
    else:
        indices = np.linspace(0, len(chunks) - 1, limit, dtype=int).tolist()
    evidence = []
    for number, index in enumerate(indices, start=1):
        chunk = chunks[index]
        evidence.append({**chunk, "source_id": f"S{number}"})
    return evidence


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("The local model did not return valid structured output. Please try again.")
        return json.loads(match.group(0))


def generate_quiz(vectorstore, question_count):
    evidence = sampled_chunks(vectorstore)
    source_map = {item["source_id"]: item for item in evidence}
    context = "\n\n".join(
        f"[{item['source_id']}] {item['source']} page {item['page']}: {item['text']}"
        for item in evidence
    )
    prompt = (
        f"Create exactly {question_count} multiple-choice study questions using only "
        "the PDF passages below. Each question must have four choices labelled A, B, "
        "C, D; one correct answer; a short explanation; a concise topic; and one or "
        "more source IDs copied exactly from the passages. Avoid trick questions. "
        "Return JSON only in this shape: {\"questions\":[{\"topic\":\"...\","
        "\"question\":\"...\",\"choices\":{\"A\":\"...\",\"B\":\"...\","
        "\"C\":\"...\",\"D\":\"...\"},\"answer\":\"A\","
        "\"explanation\":\"...\",\"sources\":[\"S1\"]}]}\n\n"
        f"PDF passages:\n{context}"
    )
    data = parse_json_response(ollama_chat(prompt, max_tokens=1800, json_mode=True))
    questions = []
    for item in data.get("questions", []):
        choices = item.get("choices", {})
        correct = str(item.get("answer", "")).upper()
        if item.get("question") and all(key in choices for key in ("A", "B", "C", "D")) and correct in choices:
            item["answer"] = correct
            item["sources"] = [key for key in item.get("sources", []) if key in source_map]
            item["source_details"] = [source_map[key] for key in item["sources"]]
            item["topic"] = str(item.get("topic", "General review"))[:80]
            questions.append(item)
    if not questions:
        raise ValueError("The quiz could not be formatted. Please generate it again.")
    return questions


def load_progress():
    empty = {"quiz_attempts": []}
    try:
        if PROGRESS_FILE.exists():
            saved = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            return {**empty, **saved}
    except (OSError, json.JSONDecodeError):
        pass
    return empty


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def record_quiz_attempt(progress, question, selected):
    correct = selected == question["answer"]
    progress["quiz_attempts"].append({
        "topic": question["topic"],
        "question": question["question"],
        "correct": correct,
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "sources": [
            {"source": item["source"], "page": item["page"]}
            for item in question.get("source_details", [])
        ],
    })
    return correct


def render_sources(items):
    if not items:
        return
    seen = set()
    for item in items:
        identity = (item.get("source"), item.get("page"))
        if identity in seen:
            continue
        seen.add(identity)
        st.caption(f"Source: {item['source']} — page {item['page']}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Your Study Help", page_icon="🤝", layout="wide")
    st.title("Your Study Help 🤝")
    st.caption("Study your PDFs with clear answers, summary, quizzes and keep track of your progress.")

    try:
        status = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        status.raise_for_status()
        installed_models = {model.get("name", "") for model in status.json().get("models", [])}
    except requests.RequestException:
        st.error("Ollama is not running. Open Ollama, then restart this app.")
        st.stop()
    missing = [model for model in (CHAT_MODEL, EMBEDDING_MODEL) if model not in installed_models]
    if missing:
        st.error("Missing Ollama models: " + ", ".join(missing) + ". In PowerShell run `ollama pull qwen3:4b` and `ollama pull embeddinggemma:300m`.")
        st.stop()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.pages = []
        st.session_state.pdf_names = []
        st.session_state.messages = []
        st.session_state.study_summary = ""
        st.session_state.quiz = []
        st.session_state.quiz_feedback = {}
        st.session_state.progress_data = load_progress()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs", disabled=not pdf_docs, use_container_width=True):
            with st.spinner("Reading PDFs and creating searchable evidence…"):
                try:
                    pages = read_pdf_pages(pdf_docs)
                    chunks = get_text_chunks(pages)
                    st.session_state.vectorstore = get_vectorstore(chunks)
                    st.session_state.pages = pages
                    st.session_state.pdf_names = [pdf.name for pdf in pdf_docs]
                    st.session_state.messages = []
                    st.session_state.study_summary = ""
                    st.session_state.quiz = []
                    st.session_state.quiz_feedback = {}
                    st.success(f"Processed {len(pdf_docs)} PDF(s), with text on {len(pages)} pages.")
                except Exception as exc:
                    st.error(f"Could not process the PDFs: {exc}")

    has_docs = st.session_state.vectorstore is not None
    ask_tab, summary_tab, quiz_tab, progress_tab = st.tabs([
        "Ask your PDFs", "Study summary", "Practice quiz", "Progress"
    ])

    with ask_tab:
        st.subheader("Ask your PDFs")
        st.caption("Answers use your documents and include the supporting PDF pages.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("evidence"):
                    render_sources(message["evidence"])
        question = st.chat_input("Ask a question about your PDFs", disabled=not has_docs)
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Finding evidence and preparing a clear answer…"):
                    try:
                        answer, evidence = answer_question(question, st.session_state.vectorstore, st.session_state.messages[:-1])
                        st.markdown(answer)
                        if evidence:
                            render_sources(evidence)
                        st.session_state.messages.append({"role": "assistant", "content": answer, "evidence": evidence})
                    except Exception as exc:
                        st.error(f"The request failed: {exc}")

    with summary_tab:
        st.subheader("Study summary")
        st.write("Create student-friendly notes from all processed PDFs, with page references.")
        if st.button("Create study summary", disabled=not st.session_state.pages):
            progress = st.progress(0, text="Preparing the summary…")
            with st.spinner("Creating study notes from your PDFs…"):
                try:
                    st.session_state.study_summary = summarize_documents(
                        st.session_state.pages,
                        st.session_state.pdf_names,
                        progress_callback=lambda done, total, name, finished: progress.progress(
                            done / total,
                            text=(f"Completed PDF {done} of {total}: {name}" if finished else f"Summarizing PDF {done + 1} of {total}: {name}…"),
                        ),
                    )
                except Exception as exc:
                    st.error(f"Could not create the study summary: {exc}")
        if st.session_state.study_summary:
            st.markdown(st.session_state.study_summary)
            st.download_button("Download study summary", st.session_state.study_summary, "study_summary.md", "text/markdown")

    with quiz_tab:
        st.subheader("Practice quiz")
        count = st.selectbox("Number of questions", [3, 5, 10], index=1, disabled=not has_docs)
        if st.button("Generate practice quiz", disabled=not has_docs):
            with st.spinner("Making questions from your PDFs…"):
                try:
                    st.session_state.quiz = generate_quiz(st.session_state.vectorstore, count)
                    st.session_state.quiz_feedback = {}
                except Exception as exc:
                    st.error(f"Could not generate the quiz: {exc}")
        for index, item in enumerate(st.session_state.quiz):
            st.markdown(f"### Question {index + 1}: {item['topic']}")
            st.write(item["question"])
            answer_key = f"quiz_choice_{index}"
            selected = st.radio(
                "Choose one answer",
                list(item["choices"].keys()),
                format_func=lambda key, choices=item["choices"]: f"{key}. {choices[key]}",
                key=answer_key,
                horizontal=False,
            )
            if st.button("Check answer", key=f"quiz_check_{index}"):
                correct = record_quiz_attempt(st.session_state.progress_data, item, selected)
                save_progress(st.session_state.progress_data)
                st.session_state.quiz_feedback[index] = {
                    "correct": correct,
                    "answer": item["answer"],
                    "explanation": item.get("explanation", ""),
                }
            feedback = st.session_state.quiz_feedback.get(index)
            if feedback:
                if feedback["correct"]:
                    st.success("Correct!")
                else:
                    st.warning(f"Not quite. The correct answer is {feedback['answer']}: {item['choices'][feedback['answer']]}")
                st.write(feedback["explanation"])
                render_sources(item.get("source_details", []))

    with progress_tab:
        st.subheader("Your study progress")
        progress_data = st.session_state.progress_data
        attempts = progress_data.get("quiz_attempts", [])
        correct_count = sum(1 for attempt in attempts if attempt.get("correct"))
        quiz_accuracy = round(100 * correct_count / len(attempts)) if attempts else 0
        col1, col2 = st.columns(2)
        col1.metric("Quiz accuracy", f"{quiz_accuracy}%")
        col2.metric("Answers submitted", len(attempts))
        st.progress(quiz_accuracy / 100, text=f"Quiz accuracy: {quiz_accuracy}%")

        if attempts:
            latest_by_question = {}
            topic_attempts = defaultdict(list)
            for attempt in attempts:
                latest_by_question[attempt["question"]] = attempt
                topic_attempts[attempt["topic"]].append(attempt)
            st.markdown("**Questions to review**")
            review_questions = [a for a in latest_by_question.values() if not a["correct"]]
            if review_questions:
                for attempt in review_questions:
                    st.write(f"- **{attempt['topic']}**: {attempt['question']}")
                    render_sources(attempt.get("sources", []))
            else:
                st.success("No questions are currently marked for review.")
            st.markdown("**Topic review**")
            for topic, topic_items in topic_attempts.items():
                last_attempt = topic_items[-1]
                status_text = "Needs more practice" if not last_attempt["correct"] else "Reviewed"
                st.write(f"- **{topic}** — {status_text} ({len(topic_items)} attempt(s))")
        else:
            st.info("Complete quiz questions to start tracking reviewed topics and questions.")


if __name__ == "__main__":
    main()
