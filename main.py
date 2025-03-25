import streamlit as st
import os
import tempfile
from pathlib import Path
import time


from llm_mechs import AIExamHelper


st.set_page_config(
    page_title="AI Exam Helper",
    page_icon="📚",
    layout="wide"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "helper" not in st.session_state:

    st.session_state.helper = AIExamHelper(
        model_path="./fine_tuned_mistral",
        db_path="./chroma_db"
    )


def save_uploaded_file(uploaded_file):

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


class StreamlitFileWrapper:


    def __init__(self, uploaded_file):
        self.name = uploaded_file.name
        self._uploaded_file = uploaded_file
        self._temp_path = None

    def save_temp(self):

        self._temp_path = save_uploaded_file(self._uploaded_file)
        return self._temp_path

    def cleanup(self):

        if self._temp_path and os.path.exists(self._temp_path):
            os.remove(self._temp_path)



with st.sidebar:
    st.title("📚 AI Exam Helper")
    st.markdown("---")


    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents or Images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents", key="process_docs"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")


                file_wrapper = StreamlitFileWrapper(file)
                temp_path = file_wrapper.save_temp()

                if temp_path:

                    class UploadedFile:
                        def __init__(self, name, path):
                            self.name = name
                            self.path = path

                        def read(self):
                            with open(self.path, 'rb') as f:
                                return f.read()



                    django_like_file = UploadedFile(file.name, temp_path)
                    result = st.session_state.helper.process_document(django_like_file)


                    os.remove(temp_path)

                    if result['success']:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ Processed document: {file.name}\n\nExtracted {result['char_count']} characters."
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Failed to process {file.name}: {result.get('error', 'Unknown error')}"
                        })


                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("All documents processed!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

    st.markdown("---")


    st.subheader("Generate Content")

    subject = st.text_input("Subject", "Physics")
    topic = st.text_input("Topic (optional)", "")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Formulas"):
            with st.spinner("Generating formulas..."):
                result = st.session_state.helper.generate_formulas_and_derivations(subject, topic)
                if result['success']:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Show me formulas and derivations for {subject} {topic}."
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['formulas_and_derivations']
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Failed to generate formulas: {result.get('error', 'Unknown error')}"
                    })

    with col2:
        difficulty = st.select_slider(
            "Question Difficulty",
            options=["easy", "medium", "hard"],
            value="medium"
        )
        count = st.slider("Number of Questions", min_value=1, max_value=10, value=5)

        if st.button("Generate Questions"):
            with st.spinner("Generating practice questions..."):
                result = st.session_state.helper.generate_practice_questions(subject, topic, difficulty, count)
                if result['success']:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Generate {count} {difficulty} practice questions for {subject} {topic}."
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['questions_and_solutions']
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Failed to generate questions: {result.get('error', 'Unknown error')}"
                    })


st.title("AI Exam Helper Chat")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me about your study materials..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    #
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.helper.answer_question(prompt)
            if result['success']:
                st.markdown(result['answer'])
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
            else:
                error_message = f"❌ I encountered an error: {result.get('error', 'Unknown error')}"
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})