# AI Angela Han - Gradio Chat Interface

This Gradio application provides a web-based chat interface to interact with an AI persona of Angela Han. It leverages Retrieval Augmented Generation (RAG) to answer questions and engage in conversations based on Angela Han's publicly available content. The AI uses Google's Gemini LLM, Google's text embeddings, and a FAISS vector store for knowledge retrieval.

## ✨ Features

*   **Interactive Chat:** Engage in conversations with the AI Angela Han persona.
*   **Angela Han Persona:** Responds as an AI version of Angela Han, drawing on her perspectives and communication style (defined in system prompts).
*   **Retrieval Augmented Generation (RAG):** Uses a FAISS vector store (loaded from a `.zip` file) to retrieve relevant information, providing contextually rich answers.
*   **Contextual Memory:** Remembers the conversation history within the current session.
*   **Streaming Responses:** Bot responses are streamed token by token for a more interactive experience.
*   **Multi-Language Capability:** The persona prompt instructs the bot to detect and respond in the language of the user's last question.
*   **Customizable UI:** Uses Gradio with some custom CSS and a custom avatar for the bot.
*   **Health Check:** Includes an optional simple HTTP server for health checks (useful for deployments like Hugging Face Spaces).
*   **Configuration:**
    *   Requires `GOOGLE_API_KEY` for LLM and embedding services.
    *   Requires `FAISS_INDEX_ZIP_PATH` pointing to the pre-built vector store.
    *   Model names (`GOOGLE_EMBEDDING_MODEL_NAME`, `GEMINI_LLM_MODEL_NAME`) are configurable via environment variables.

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Gradio:** For creating the web UI.
*   **LangChain:** Framework for LLM application development.
    *   `langchain_google_genai`: For Google Gemini LLM and Embeddings.
    *   `langchain_community.vectorstores.FAISS`: For FAISS vector store.
    *   Various chain components (`create_retrieval_chain`, `create_history_aware_retriever`, etc.).
*   **Google Generative AI:**
    *   LLM Model: `gemini-2.0-flash-lite` (default, configurable).
    *   Embedding Model: `models/text-embedding-004` (default, configurable).
*   **FAISS (faiss-cpu/faiss-gpu):** For efficient similarity search in the vector store.
*   **python-dotenv:** For managing environment variables locally.
*   **Standard Python Libraries:** `os`, `asyncio`, `zipfile`, `tempfile`, `shutil`, `logging`, `http.server`, `threading`, `base64`.

## ⚙️ Prerequisites

1.  **Python 3.8 or higher.**
2.  **Google API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.
    *   Create an API key.
    *   **Enable the "Generative Language API"** (also known as Gemini API) for your project.
    *   This key needs to be available as an environment variable (`GOOGLE_API_KEY`). When deploying to Hugging Face Spaces, set this as a "Secret".
3.  **FAISS Index ZIP File:**
    *   You need a `.zip` file (e.g., `faiss_index_google.zip`, configurable via `FAISS_INDEX_ZIP_PATH`).
    *   This ZIP file must contain `index.faiss` and `index.pkl` files.
    *   These files should be generated from your knowledge base documents using the same `GoogleGenerativeAIEmbeddings` model specified in this application (default: `models/text-embedding-004`).
    *   You can use the embedding_generator (https://github.com/rurounigit/embedding_generator/) to create this file.
4.  **SVG Avatar (Optional):**
    *   The app looks for `pink_dot_avatar.svg` in the same directory as `app.py` for the bot's avatar. You can replace this or remove the reference.

## 🚀 Setup & Installation (Local Development)

1.  **Clone the repository (if applicable) or download `app.py`, `requirements.txt`, and `pink_dot_avatar.svg` (if using custom avatar):**
    ```bash
    # If in a repo:
    # git clone https://github.com/rurounigit/huggingface_bot_gradio.git
    # cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have `app.py` and `requirements.txt` in your current directory.
    ```bash
    pip install -r requirements.txt
    ```
    *(The provided `requirements.txt` includes `gradio`, `langchain` (and related packages), `faiss-cpu`, `python-dotenv`, and `tiktoken`.)*

4.  **Prepare the FAISS Index:**
    *   Ensure your `faiss_index_google.zip` (or the file specified by `FAISS_INDEX_ZIP_PATH`) is present in the root directory or the specified path.

5.  **Configure Environment Variables (Local):**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    FAISS_INDEX_ZIP_PATH="faiss_index_google.zip" # Or the path to your FAISS index ZIP
    # Optional: Override default model names if needed
    # GOOGLE_EMBEDDING_MODEL_NAME="models/text-embedding-004"
    # GEMINI_LLM_MODEL_NAME="gemini-2.0-flash-lite"
    ```

## ▶️ Running the App

### Locally
To run the Gradio app locally:
```bash
python app.py
```
The application will typically be available at `http://0.0.0.0:7860` (Gradio will print the exact URL). A dummy health check server will also start on port `7861`.

### On Hugging Face Spaces
1.  Create a new Space on Hugging Face.
2.  Choose "Gradio" as the SDK.
3.  Upload your `app.py`, `requirements.txt`, your FAISS index zip file (e.g., `faiss_index_google.zip`), and `pink_dot_avatar.svg` (if using).
4.  Go to your Space's "Settings" tab.
5.  Under "Repository secrets," add new secrets:
    *   **Name:** `GOOGLE_API_KEY`
      *   **Value:** Your actual Google API Key
    *   **(Optional) Name:** `FAISS_INDEX_ZIP_PATH`
      *   **Value:** The name of your uploaded FAISS index zip file (e.g., `faiss_index_google.zip`). If not set, it defaults to `faiss_index_google.zip`.
    *   **(Optional) Name:** `GOOGLE_EMBEDDING_MODEL_NAME`
      *   **Value:** Your desired embedding model (e.g., `models/text-embedding-004`)
    *   **(Optional) Name:** `GEMINI_LLM_MODEL_NAME`
      *   **Value:** Your desired LLM model (e.g., `gemini-2.0-flash-lite`)
6.  The Space should build and launch your Gradio app. Ensure your FAISS index zip path correctly points to the file within the Space's file system.

## 💬 Usage

1.  **Open the Gradio App:** Access it via the local URL or your Hugging Face Space URL.
2.  **Chat:** Type your message in the textbox at the bottom and press Enter or click the send button.
3.  The AI Angela Han will respond, drawing information from its knowledge base and maintaining conversation context.

## 🧠 FAISS Index (Knowledge Base)

*   **Source:** The bot relies on a pre-built FAISS index provided via `FAISS_INDEX_ZIP_PATH`.
*   **Contents:** The ZIP file must contain `index.faiss` and `index.pkl`.
*   **Embeddings:** This index **must** be created using the same Google Generative AI embedding model configured in `app.py` (default: `models/text-embedding-004`).
*   **Creation:** This app *loads* the index. You need a separate process or tool (like the [companion Document Embedding Generator](link_to_your_other_generator_app_readme_or_repo_if_applicable)) to create it.

## 🎨 Persona & Prompts

The bot's persona and response logic are heavily influenced by:

*   `contextualize_q_system_prompt`: Rewrites user questions to be standalone based on chat history.
*   `persona_qa_system_prompt`: Defines Angela Han's persona, instructs on language use, and how to integrate retrieved context with chat history.

These prompts are located in `app.py` and can be modified to alter the bot's behavior.

## ⚠️ Important Notes

*   **`GOOGLE_API_KEY`:** Essential for functionality.
*   **`FAISS_INDEX_ZIP_PATH`:** The app will fail to load the knowledge base if this file is missing or incorrectly specified.
*   **Model Consistency:** Ensure the embedding model used to create the FAISS index matches the `GOOGLE_EMBEDDING_MODEL_NAME` used by this app.
*   **Resource Usage:** The LLM and embedding models are API-based, but loading the FAISS index can consume memory.
*   **Disclaimer:** The app includes a disclaimer that it's a fan AI and not endorsed by Angela Han.

## 🔧 Troubleshooting

*   **"CRITICAL: GOOGLE_API_KEY environment variable not found"**:
    *   Verify your `GOOGLE_API_KEY` is correct and set as an environment variable (local) or Space secret (Hugging Face).
    *   Ensure the Generative Language API is enabled for your Google Cloud project.
*   **"CRITICAL: RAG chain initialization failed"**:
    *   **FAISS Index Issues:**
        *   Check that `FAISS_INDEX_ZIP_PATH` points to the correct file and the file exists.
        *   Ensure the zip file contains `index.faiss` and `index.pkl`.
        *   Verify the index was created with the *exact same* embedding model (`GOOGLE_EMBEDDING_MODEL_NAME`).
    *   **LLM/Embedding Model Issues:** Check logs for errors related to `ChatGoogleGenerativeAI` or `GoogleGenerativeAIEmbeddings` initialization (e.g., invalid API key, model name typos).
    *   Look at the application logs (console or Hugging Face Space logs) for more detailed error messages.
*   **Bot says "Sorry, I'm not fully initialized yet."**: This usually means the RAG chain failed to set up. Check the logs for earlier errors.
*   **Avatar not showing:** Ensure `pink_dot_avatar.svg` is in the same directory as `app.py` or update the path in `gr.Chatbot`.
*   **Hugging Face Space Build/Runtime Failures:**
    *   Double-check `requirements.txt` for correct package names and versions.
    *   Ensure all necessary files (app.py, requirements.txt, FAISS zip, avatar SVG) are uploaded to the Space.
    *   Verify secret names and values in Space settings.

## 📄 License

Apache 2.0

---
title: Chatbot
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
short_description: chatbot
---
