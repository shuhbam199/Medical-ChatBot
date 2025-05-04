# chatbot
 
# 🩺 MediBot — LLM-Powered Medical Chatbot with RAG

MediBot is an intelligent, document-aware medical chatbot built using **LangChain**, **FAISS**, **HuggingFace Mistral-7B**, and **Streamlit**. It uses **Retrieval-Augmented Generation (RAG)** to answer questions based **only on trusted medical PDFs**.

> ⚠️ This project is for **educational and informational purposes only**. It is **not intended for clinical use** or to replace professional medical advice.

---

## 🔍 Features

- 🧠 **LLM-backed**: Uses Mistral-7B-Instruct hosted on HuggingFace
- 📄 **Document-aware**: Answers are grounded in uploaded medical PDFs
- 🧾 **Retrieval-augmented**: FAISS vector store enables context-based answers
- 🧑‍💻 **Interactive UI**: Built with Streamlit for real-time chat
- 📚 **Transparent**: View source text chunks used to generate answers
- 💬 **Context-controlled**: Custom prompts prevent hallucinations

---

## 📁 Project Structure
```User Query │ ▼ FAISS Retriever │ ▼ Relevant Context │ ▼ Mistral LLM │ ▼ Answer ▲ │ Document Embeddings (via sentence-transformers)```

```medibot/ ├── create_memory.py # Preprocess PDFs and create FAISS vector store ├── connect_with_llm.py # CLI-based testing (optional) ├── medibot.py # Streamlit chat interface ├── vectorstore/ # Stores FAISS index (auto-generated) ├── data/ # Folder for your medical PDFs ├── .env # HuggingFace API token └── requirements.txt # Project dependencies```

2. Install Requirement
   ```pip install -r requirements.txt```
3. Add Your HuggingFace Token
  Create a .env file in the root folder:
```HF_TOKEN=your_huggingface_token_here```
You can get a token from: https://huggingface.co/settings/tokens

4. Add Your PDF Files
Place trusted medical documents (e.g. drug guides, disease handbooks) into the data/ folder.

5. Build the Knowledge Base
```python create_memory.py```

6. Connect with LLM
   
8. Run the Chatbot


🙋‍♂️ Author
Shubham Dange
Data Scientist & LLM Developer
GitHub

