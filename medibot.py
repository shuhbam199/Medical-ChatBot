import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

def main():
    st.title("Ask Chatbot!")
    st.write("Ready to take your query.")
    st.chat_input("Type something...")  # Just for test

# Optional dotenv for local testing
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )

def main():
    st.title("Ask Chatbot!")
    st.write("✅ UI loaded")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'], unsafe_allow_html=True)

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don’t know the answer, just say that you don’t know — don’t try to make up an answer.
        Don’t provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "")
            source_documents = response.get("source_documents", [])

            # Compose assistant reply
            answer_html = f"<p><b>Answer:</b> {result}</p>"

            # Only show source docs if relevant and not a "don't know" or unrelated reply
            show_sources = (
                source_documents
                and "don't know" not in result.lower()
                and "does not provide" not in result.lower()
            )

            source_html = ""
            if show_sources:
                source_html += "<hr><p><b>Source Docs:</b></p>"
                for doc in source_documents:
                    snippet = doc.page_content.strip().replace("\n", " ")
                    snippet = snippet[:400] + "..." if len(snippet) > 400 else snippet
                    source_html += f"<div style='color:gray; font-size: 0.85rem; margin-top: 0.5rem;'>{snippet}</div>"

            combined_html = answer_html + source_html
            st.chat_message('assistant').markdown(combined_html, unsafe_allow_html=True)

            # Store clean version in chat history
            st.session_state.messages.append({'role': 'assistant', 'content': combined_html})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
