"""AI Assistant / Talk-to-Data Chatbot page."""

from __future__ import annotations

import streamlit as st

from src.database.database import is_applicants_table_populated, load_applicants_table
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from src.talk_to_data.conversation_memory import ConversationMemory
from src.talk_to_data.nl_to_sql import ask_question


def _ensure_database_populated() -> bool:
    if is_applicants_table_populated():
        return True

    try:
        df = load_raw_data()
        load_applicants_table(df)
        return True

    except (DatasetNotFoundError, DatasetValidationError) as exc:
        st.error(str(exc))
        return False


def render() -> None:
    # Page header
    header_col, button_col = st.columns([5, 1])

    with header_col:
        st.header("AI Assistant — Talk to Your Data")
        st.caption(
            "Ask questions about the applicant dataset in plain English. "
            "Get data-driven answers based on the current dataset."
        )

    if not _ensure_database_populated():
        return

    # Initialize chat state
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = ConversationMemory()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Clear conversation button
    with button_col:
        st.write("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["chat_memory"].clear()
            st.rerun()

    # Example questions
    with st.expander("Example questions you can ask"):
        st.markdown(
            "- What is the average income of applicants?\n"
            "- How many applicants have children?\n"
            "- Which income category has the highest default rate?\n"
            "- Compare the default rate between employed and unemployed applicants.\n"
            "- Which occupation groups have the highest default rate?\n"
            "- What is the average credit amount?\n"
            "- What is the overall default rate?"
        )

    # Display previous conversation
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            if turn.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(turn["sql"], language="sql")

    # Chat input
    question = st.chat_input("Ask a question about the applicant data...")

    if not question:
        return

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the data..."):
            response = ask_question(
                question,
                memory=st.session_state["chat_memory"],
            )

        st.write(response.answer)

        if response.generated_sql:
            with st.expander("Generated SQL"):
                st.code(response.generated_sql, language="sql")

        if response.validation_status == "rejected":
            st.caption(
                "⚠️ The generated query could not be safely processed. "
                "Please try rephrasing your question."
            )

    # Save conversation
    st.session_state["chat_history"].append(
        {
            "question": question,
            "answer": response.answer,
            "sql": response.generated_sql,
        }
    )