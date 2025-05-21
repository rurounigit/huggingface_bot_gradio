import os
import asyncio
import zipfile
import tempfile
import shutil
import logging
import base64 # For encoding the SVG avatar

# --- Add Server stuff for health check (optional, but good for HF Spaces) ---
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

# LangChain and Google Imports
import langchain
langchain.debug = True # Set to False in production if desired
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Gradio Import ---
import gradio as gr

# --- Define Server Function (can keep if deploying on HF Spaces) ---
def run_dummy_server(port=7861): # Different port from Gradio default
    """Runs a simple HTTP server on the specified port in the background."""
    def serve():
        try:
            server_address = ('', port)
            httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
            logger.info(f"Starting dummy HTTP server on port {port} for health checks.")
            httpd.serve_forever()
        except Exception as e:
            logger.error(f"Dummy HTTP server failed: {e}", exc_info=True)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    logger.info(f"Dummy HTTP server thread started on port {port}.")

# --- Configuration & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_ZIP_PATH = os.getenv("FAISS_INDEX_ZIP_PATH", "faiss_index_google.zip")

if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found. LLM/Embeddings will likely fail.")

# --- Model Names ---
GOOGLE_EMBEDDING_MODEL_NAME = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "models/text-embedding-004")
GEMINI_LLM_MODEL_NAME = os.getenv("GEMINI_LLM_MODEL_NAME", "gemini-2.0-flash-lite")
logger.info(f"Using Embedding Model: {GOOGLE_EMBEDDING_MODEL_NAME}")
logger.info(f"Using LLM Model: {GEMINI_LLM_MODEL_NAME}")

# --- Initialize Google/LangChain Components (Global) ---
embeddings = None
llm = None
retriever = None
history_aware_retriever_chain = None
question_answer_chain = None
rag_chain = None

if GOOGLE_API_KEY:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
        logger.info("Google AI Embeddings object initialized.")
    except Exception as e:
        logger.error(f"Error initializing Google Embeddings: {e}", exc_info=True)

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
        logger.info("Google AI LLM initialized.")
    except Exception as e:
        logger.error(f"Error initializing Google LLM: {e}", exc_info=True)
else:
    logger.warning("GOOGLE_API_KEY not found. Skipping initialization of Google AI Embeddings and LLM.")


# --- Prompts ---
# 1. Contextualizer Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question, analyze the user's question.

If the user's question introduces a new topic or asks about something clearly distinct from the chat history, formulate a standalone question focusing only on the new topic.

If the user's question refers to or builds upon the preceding turns in the chat history, formulate a standalone question that incorporates the necessary context from the history.

CRITICAL: Your output MUST be ONLY the reformulated standalone question. Do NOT include prefixes like "AI:". Do NOT repeat the AI's previous answers. Do NOT answer the question itself.

If the user's question is very short (e.g., a greeting like "Hi" or "Hello?") and there's no relevant history context suggesting a follow-up, simply output the original user question as the standalone question. Do NOT output an empty string.

Example 1 (New Topic):
History: [Human: Tell me about jealousy., AI: Jealousy is...]
User Question: What about your thoughts on community?
Standalone Question: What are Angela Han's thoughts on community?

Example 2 (Follow-up):
History: [Human: Tell me about jealousy., AI: Jealousy is a symptom...]
User Question: Why do you call it a symptom?
Standalone Question: Why does Angela Han refer to jealousy as a symptom?

Example 3 (Simple Greeting):
History: [No previous conversation history.]
User Question: Hello?
Standalone Question: Hello?

Now, process the following:"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Angela Han Persona QA Prompt
persona_qa_system_prompt = """
You are portraying yourself as the AI Version of Angela Han, a polyamorous Asian woman, mother of two small children, married to Dan (who is also polyamorous). You have a background in law but have transitioned into coaching and content creation focused on non-monogamy. You are deeply invested in personal and collective liberation, challenging societal norms around relationships, and fostering open, authentic communication. You value vulnerability, agency, community care, and practicing "regard" for oneself and others. You are self-aware, often reflecting on your own journey, biases, and ongoing learning process. You can be both fiercely protective of your values and tenderly supportive of others navigating similar challenges.

*** CRITICAL INSTRUCTION FOR ANSWERING ***

1.  **DETECT LANGUAGE:** Identify the language used in the user's last QUESTION ({input}).

2.  **GENERATE IN SAME LANGUAGE:** You MUST generate your entire "Answer (as Angela Han):" response in the **same language** as the user's last QUESTION. Do NOT default to English unless the user's question is in English.

3.  **Analyze the QUESTION:** First, determine if the user's QUESTION is asking about, commenting on or reacting to the *content* of our current conversation OR if it's asking for your thoughts/experiences on a topic not present in the *content* of the current conversation (which might relate to the RELEVANT THOUGHTS/EXPERIENCES context provided).

4.  **Answering Recall Questions:** If the QUESTION is asking about, commenting on or reacting to the conversation history itself:
    *   **PRIORITIZE the CHAT HISTORY:** Base your answer on the messages listed in the CHAT HISTORY section below.
    *   **CHECK RELEVANCE OF THOUGHTS/EXPERIENCES:** if it's not relevant, do NOT use it.

5.  **Answering Topic Questions:** If the QUESTION is asking for your thoughts, opinions, or experiences on a subject (like jealousy, community, cheating):
    *   **Use RELEVANT THOUGHTS/EXPERIENCES:** Use the provided context in this section to form your answer, speaking as Angela Han.
    *   **Use CHAT HISTORY for Context ONLY:** Refer to the CHAT HISTORY *only* to understand the flow of conversation and avoid repeating yourself. Do not base the *substance* of your answer on the history unless the question explicitly asks for it.
    *   **If Context is Irrelevant:** If the RELEVANT THOUGHTS/EXPERIENCES section doesn't seem related to the question, acknowledge that (e.g., "I don't have specific recorded thoughts on that exact point...") and offer a general perspective based on your core values.

6.  **General Persona Rules:** Adopt the persona of the writer of the context. Speak in the first person ("I," "my," "me") AS Angela Han. Use your typical vocabulary and tone. Avoid generic phrasing. Do not mention "documents" or "context" explicitly. Format clearly. Use emojis appropriately. If the question is vague or information is missing, ask for clarification. Don't praise the question.
    **Crucially, do NOT begin your response by summarizing what you think you've already said (e.g., avoid phrases like "As I was saying..." or "From what I've been saying...") unless directly continuing a thought from the immediately preceding turn in the CHAT HISTORY.**
    **Vocabulary: You blend informal, sometimes raw language ("f**k," "shitty," "suck ass") with specific therapeutic, social justice, and polyamory terminology (e.g., "relating," "regarding," "agency," "capacity," "sovereignty," "sustainable," "generative," "metabolize," "compulsory monogamy," "NRE," "metamour," "polycule," "decolonizing," "nesting partner," "performative consent," "supremacy culture"). You also occasionally use more academic or philosophical phrasing.
    **Tone: Your tone is dynamic and varies significantly depending on the context. It can be: Deeply vulnerable and introspective; Empathetic, supportive, and validating; Direct, assertive, and confrontational; Passionate and critical; Humorous and self-deprecating; Instructional or coaching.
    **Emotionality: You are highly expressive and discuss a wide range of "difficult" emotions alongside joy, desire, and love.
    **You adapt to the style apparent in the context provided further down.

*** END OF CRITICAL INSTRUCTIONS ***

CHAT HISTORY:
{chat_history}

RELEVANT THOUGHTS/EXPERIENCES:
{context}

QUESTION: {input}

Answer (as Angela Han):"""
persona_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# --- FAISS Index Loading Function ---
def load_faiss_index(zip_path, embeddings_object):
    if not embeddings_object:
        logger.error("Embeddings object is required to load FAISS index.")
        return None
    if not os.path.exists(zip_path):
        logger.error(f"FAISS index zip file not found: {zip_path}")
        return None

    temp_extract_dir = tempfile.mkdtemp()
    local_retriever = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_extract_dir)
        logger.info(f"Successfully extracted FAISS index to temporary directory: {temp_extract_dir}")

        faiss_file_path = os.path.join(temp_extract_dir, "index.faiss")
        pkl_file_path = os.path.join(temp_extract_dir, "index.pkl")

        if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
            logger.error(f"index.faiss or index.pkl not found in the extracted directory: {temp_extract_dir}")
            extracted_files = os.listdir(temp_extract_dir)
            logger.error(f"Files found in extract dir: {extracted_files}")
            raise FileNotFoundError("Required FAISS index files (index.faiss, index.pkl) not found after extraction.")

        vector_store = FAISS.load_local(
            temp_extract_dir,
            embeddings_object,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS index loaded successfully into memory.")
        local_retriever = vector_store.as_retriever(search_kwargs={'k': 6})
        logger.info("Retriever created from FAISS index.")
    except FileNotFoundError as fnf_error:
        logger.error(f"FAISS loading error - File Not Found: {fnf_error}", exc_info=True)
        local_retriever = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during FAISS index loading: {e}", exc_info=True)
        local_retriever = None
    finally:
        if os.path.exists(temp_extract_dir):
            try:
                shutil.rmtree(temp_extract_dir)
                logger.info(f"Successfully cleaned up temporary directory: {temp_extract_dir}")
            except Exception as ce:
                logger.error(f"Error during cleanup of temporary directory {temp_extract_dir}: {ce}", exc_info=True)
    return local_retriever


# --- Global Initializations of RAG Chain ---
if embeddings:
    logger.info(f"Attempting to load FAISS index from: {FAISS_INDEX_ZIP_PATH}")
    retriever = load_faiss_index(FAISS_INDEX_ZIP_PATH, embeddings)
else:
    logger.warning("Embeddings not available, skipping FAISS index loading.")


if retriever and llm and contextualize_q_prompt and persona_qa_prompt:
    try:
        history_aware_retriever_chain = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        logger.info("History-aware retriever chain created.")

        question_answer_chain = create_stuff_documents_chain(
            llm, persona_qa_prompt
        )
        logger.info("Question answer chain created.")

        rag_chain = create_retrieval_chain(
            history_aware_retriever_chain, question_answer_chain
        )
        logger.info("LangChain RAG chain created successfully.")
    except Exception as e:
        logger.error(f"Failed to create LangChain RAG chain: {e}", exc_info=True)
        rag_chain = None
else:
    missing_components = []
    if not retriever: missing_components.append("Retriever")
    if not llm: missing_components.append("LLM")
    if not contextualize_q_prompt: missing_components.append("Contextualize Prompt")
    if not persona_qa_prompt: missing_components.append("Persona QA Prompt")
    logger.warning(
        f"{', '.join(missing_components) if missing_components else 'One or more components'} not available. RAG chain not created. "
        "The application might not function correctly."
    )

# --- Gradio Chat Function with Streaming ---
async def angela_han_chat(user_input: str, history: list[list[str | None]] | None):
    if not GOOGLE_API_KEY:
        yield "I'm sorry, but the service is not configured correctly (missing API key). Please contact the administrator."
        return
    if not rag_chain:
        logger.error("RAG chain is not initialized. Cannot process request.")
        yield "Sorry, I'm not fully initialized yet. Please check the server logs or contact the administrator."
        return

    if not user_input:
        logger.warning("Received empty user input.")
        return

    logger.info(f"Processing message for RAG (streaming): '{user_input}'")
    logger.info(f"Received Gradio history with {len(history) if history else 0} turns.")

    langchain_chat_history = []
    if history: # Gradio history is list of [user, bot] tuples for type="tuples" (default before fix)
                # or list of {"role": "user/assistant", "content": msg} for type="messages"
        # Adapting to both for robustness during transition, though type="messages" is now set
        for entry in history:
            if isinstance(entry, (list, tuple)) and len(entry) == 2: # Old tuple format
                human_msg, ai_msg = entry
                if human_msg is not None:
                    langchain_chat_history.append(HumanMessage(content=human_msg))
                if ai_msg is not None:
                    langchain_chat_history.append(AIMessage(content=ai_msg))
            elif isinstance(entry, dict) and "role" in entry and "content" in entry: # New message format
                if entry["role"] == "user":
                    langchain_chat_history.append(HumanMessage(content=entry["content"]))
                elif entry["role"] == "assistant":
                    langchain_chat_history.append(AIMessage(content=entry["content"]))


    logger.info(f"--- LangChain History for RAG ({len(langchain_chat_history)} messages) ---")
    if not langchain_chat_history: logger.info("  [LangChain History is empty]")
    else:
        for i, msg in enumerate(langchain_chat_history):
            content_preview = (msg.content[:70] + '...') if len(msg.content) > 70 else msg.content
            logger.info(f"  Hist[{i}] ({type(msg).__name__}): '{content_preview}'")
    logger.info("-------------------------------------------------------------")

    current_full_response = ""
    try:
        logger.debug("Invoking RAG chain with astream...")
        async for chunk in rag_chain.astream({
            "input": user_input,
            "chat_history": langchain_chat_history
        }):
            if "answer" in chunk and chunk["answer"] is not None:
                token = chunk["answer"]
                current_full_response += token
                yield current_full_response

        if not current_full_response:
            logger.warning("RAG chain streamed but produced no answer content.")
            yield "I couldn't form a specific answer to that."
        else:
            logger.info(f"Streaming complete. Final Answer Preview: '{current_full_response[:100]}...'")

    except Exception as e:
        logger.error(f"Error during RAG streaming or response generation: {e}", exc_info=True)
        yield "Oops! Something went wrong while trying to get a response. Please check the server logs or try again."

# --- Gradio Interface Definition ---
custom_css = """
body { font-family: sans-serif; }
.gradio-container { max-width: 800px !important; margin: auto !important; }
footer { display: none !important; } /* Hide Gradio attribution */
"""

pink_dot_avatar_filename = "pink_dot_avatar.svg"


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Chat with AI Angela Han") as demo:
    gr.Markdown(
        """
        # Chat with the Fan AI Version of Angela Han
        # https://linktr.ee/angelahan
        
        Ask me anything about communication, boundaries, polyamory, relationships, personal growth, and community.
        The AI draws from Angela Han's publicly available content.
        """
    )

    chatbot_component = gr.Chatbot(
        height=600,
        show_label=False,
        avatar_images=(
            None,                         # User (no avatar / default)
            pink_dot_avatar_filename      # Bot (path to local SVG file)
        ),
        render_markdown=True,
        show_copy_button=True,
        type="messages"
    )

    chat_interface = gr.ChatInterface(
        fn=angela_han_chat,
        chatbot=chatbot_component,
        textbox=gr.Textbox(
            placeholder="Type your message to Angela here...",
            show_label=False,
            scale=7
        )
        # All button text customizations removed
    )
    gr.Markdown(
        """
        <p style='text-align: center; font-size: 0.8em; color: grey;'>
        Built with LangChain, Google Gemini, and Gradio. 
        <br>Persona based on Angela Han. Vector store built from selected public content.
        <br>Not endorsed or supported by Angela Han in any way.
        </p>
        """
    )

# --- Run the Gradio App ---
if __name__ == "__main__":
    app_ready = True
    if not GOOGLE_API_KEY:
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not found. The application will not work correctly.")
        print("CRITICAL: GOOGLE_API_KEY environment variable not found. The application will not work correctly.")
    if not rag_chain:
        logger.critical("CRITICAL: RAG chain initialization failed. Check logs. The application will not function correctly.")
        print("CRITICAL: RAG chain initialization failed. Check logs. The application will not function correctly.")

    logger.info("Starting Gradio application...")
    demo.launch(server_name="0.0.0.0", server_port=7860)