import os
import certifi
import tempfile

os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

from dotenv import load_dotenv
load_dotenv()


#  Page config 
st.set_page_config(
    page_title="LangChain Search Agent",
    page_icon="🔎",
    layout="wide",
)


#  Session-state init 
for key, default in [
    ("messages", None),
    ("url_sources", {}),
    ("pdf_sources", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.messages is None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can search the web, look up research papers, Wikipedia, "
                "answer questions from URLs you add, or read PDFs you upload.\n\n"
                "Use the sidebar to add sources, then ask me anything. "
                "I'll remember what we've discussed so you can ask follow-up questions naturally!"
            ),
        }
    ]


#  Chat-history builder 
def build_chat_history(messages: list, max_turns: int = 6) -> str:
    """
    Return the last `max_turns` human/assistant exchanges as a plain-text block
    (excluding the current user message, which is handled separately).

    Only assistant messages that have actual content are included;
    error messages and the initial greeting are skipped.
    """
    # Collect only real user↔assistant pairs (skip the very first greeting)
    pairs = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        # skip empty or error-prefixed messages
        if not content or content.startswith("⚠️"):
            continue
        pairs.append((role, content))

    # Take the last max_turns messages (not counting current user msg)
    recent = pairs[-(max_turns):]  # up to last N messages

    if not recent:
        return ""

    lines = []
    for role, content in recent:
        label = "Human" if role == "user" else "Assistant"
        # Truncate very long assistant answers to avoid bloating the prompt
        if role == "assistant" and len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


def augment_prompt_with_history(prompt: str, history_text: str) -> str:
    """
    Wrap the user's new question with conversation context so the agent
    can resolve pronouns / references like 'name all 3 of them'.
    """
    if not history_text:
        return prompt

    return (
        f"[CONVERSATION HISTORY]\n"
        f"{history_text}\n\n"
        f"[CURRENT QUESTION]\n"
        f"{prompt}\n\n"
        f"Note: Use the conversation history above to resolve any references "
        f"(e.g. 'them', 'it', 'those', 'the above') before deciding which tool to call. "
        f"If the current question is a follow-up that can be answered from the history "
        f"alone, you may answer directly without calling a tool."
    )


#  Base tools ─
@st.cache_resource
def build_base_tools():
    arxiv  = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400))
    wiki   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400))
    search = DuckDuckGoSearchRun(name="DuckDuckGo_Search")
    return [search, arxiv, wiki]


#  URL / PDF tool builders 
def _make_retriever_tool(docs, tool_name: str, description: str):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb   = FAISS.from_documents(chunks, embeddings)
    retriever  = vectordb.as_retriever(search_kwargs={"k": 4})
    return create_retriever_tool(retriever, tool_name, description)


def load_url_tool(url: str, index: int):
    docs = WebBaseLoader(url).load()
    if not docs:
        return None, ""

    page_title   = docs[0].metadata.get("title", url).strip()
    body_preview = docs[0].page_content[:300].replace("\n", " ").strip()
    tool_name    = f"URL_Source_{index}"

    description = (
        f"Source type: WEBPAGE  |  Tool name: {tool_name}\n"
        f"URL: {url}\n"
        f"Page title: '{page_title}'\n"
        f"Content preview: {body_preview}\n\n"
        f"Use this tool to answer any question about the content of the above webpage. "
        f"ALWAYS prefer this tool over Wikipedia or web search when the question "
        f"matches this page's topic."
    )
    tool = _make_retriever_tool(docs, tool_name, description)
    return tool, page_title


def load_pdf_tool(file_bytes: bytes, filename: str, index: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    if not docs:
        return None

    tool_name    = f"PDF_Source_{index}"
    body_preview = docs[0].page_content[:300].replace("\n", " ").strip()

    description = (
        f"Source type: PDF  |  Tool name: {tool_name}\n"
        f"File name: '{filename}'\n"
        f"Content preview: {body_preview}\n\n"
        f"Use this tool to answer any question about the content of the uploaded "
        f"PDF '{filename}'. Always prefer this tool when the question is about "
        f"this document's subject matter."
    )
    return _make_retriever_tool(docs, tool_name, description)


#  Trace rendering helpers 
def _tool_icon(name: str) -> str:
    if name == "DuckDuckGo_Search": return "🔍"
    if name == "arxiv":             return "📄"
    if name == "wikipedia":         return "📖"
    if name.startswith("URL_"):     return "🌐"
    if name.startswith("PDF_"):     return "📕"
    return "🔧"


def _render_trace(steps: list, source_map: dict, collapsed: bool = False):
    with st.expander(
        f"📋 Agent reasoning — {len(steps)} tool call(s)",
        expanded=not collapsed,
    ):
        h = st.columns([0.5, 2, 3, 4])
        h[0].markdown("**#**")
        h[1].markdown("**Tool / Source**")
        h[2].markdown("**Query sent**")
        h[3].markdown("**Result snippet**")
        st.markdown("---")

        for i, (action, observation) in enumerate(steps, 1):
            icon    = _tool_icon(action.tool)
            label   = source_map.get(action.tool, action.tool)
            obs_str = str(observation)
            snippet = obs_str[:200].replace("\n", " ") + ("…" if len(obs_str) > 200 else "")

            row = st.columns([0.5, 2, 3, 4])
            row[0].markdown(f"**{i}**")
            row[1].markdown(f"{icon} `{action.tool}`\n\n_{label}_")
            row[2].caption(str(action.tool_input)[:120])
            row[3].caption(snippet)

        st.markdown("---")
        st.markdown("##### Full step details")
        for i, (action, observation) in enumerate(steps, 1):
            icon  = _tool_icon(action.tool)
            label = source_map.get(action.tool, action.tool)
            with st.expander(f"Step {i} — {icon} `{action.tool}`  ·  {label}", expanded=False):
                if action.log:
                    thought_lines = [
                        ln for ln in action.log.strip().splitlines()
                        if ln.startswith("Thought:")
                    ]
                    if thought_lines:
                        st.info(f"💭 {thought_lines[0]}")

                st.markdown("**Input sent to tool:**")
                st.code(str(action.tool_input), language="text")

                st.markdown("**Raw output from tool (observation):**")
                obs_text = str(observation)
                st.text(obs_text[:1500] + ("…" if len(obs_text) > 1500 else ""))


#  Source-map helper 
def _build_source_map() -> dict:
    m = {
        "DuckDuckGo_Search": "DuckDuckGo web search",
        "arxiv":             "Arxiv research papers",
        "wikipedia":         "Wikipedia",
    }
    for url, info in st.session_state.url_sources.items():
        m[info["tool_name"]] = f"{info['title']} ({url[:50]})"
    for fname, info in st.session_state.pdf_sources.items():
        m[info["tool_name"]] = f"PDF: {fname}"
    return m


#  Sidebar ─
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    # ── Chat-history settings ─
    st.divider()
    st.markdown("### 🧠 Memory Settings")
    memory_turns = st.slider(
        "Remember last N messages",
        min_value=2,
        max_value=20,
        value=6,
        step=2,
        help="How many previous messages the agent considers when answering follow-up questions.",
    )

    st.divider()

    # ── URL section 
    st.markdown("### 🌐 Add URLs")
    new_url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
        key="new_url_input",
        label_visibility="collapsed",
    )
    if st.button("➕ Add URL", use_container_width=True):
        url = new_url.strip()
        if not url:
            st.warning("Please enter a URL first.")
        elif url in st.session_state.url_sources:
            st.warning("This URL is already loaded.")
        else:
            with st.spinner(f"Loading {url} …"):
                idx = len(st.session_state.url_sources) + 1
                try:
                    tool, title = load_url_tool(url, idx)
                    if tool:
                        st.session_state.url_sources[url] = {
                            "tool":      tool,
                            "title":     title or url,
                            "tool_name": f"URL_Source_{idx}",
                        }
                        st.success(f"Loaded: {title or url}")
                    else:
                        st.error("Could not load content from that URL.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.url_sources:
        st.markdown("**Loaded URLs:**")
        for url, info in list(st.session_state.url_sources.items()):
            c1, c2 = st.columns([4, 1])
            c1.caption(f"🌐 {info['tool_name']} — {info['title'][:40]}")
            if c2.button("✕", key=f"rm_url_{url}"):
                del st.session_state.url_sources[url]
                st.rerun()

    st.divider()

    # ── PDF section 
    st.markdown("### 📕 Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_pdfs:
        for pdf_file in uploaded_pdfs:
            fname = pdf_file.name
            if fname not in st.session_state.pdf_sources:
                with st.spinner(f"Embedding {fname} …"):
                    idx = len(st.session_state.pdf_sources) + 1
                    try:
                        tool = load_pdf_tool(pdf_file.getvalue(), fname, idx)
                        if tool:
                            st.session_state.pdf_sources[fname] = {
                                "tool":      tool,
                                "tool_name": f"PDF_Source_{idx}",
                            }
                            st.success(f"Loaded: {fname}")
                    except Exception as e:
                        st.error(f"Error loading {fname}: {e}")

    if st.session_state.pdf_sources:
        st.markdown("**Loaded PDFs:**")
        for fname, info in st.session_state.pdf_sources.items():
            st.caption(f"📕 {info['tool_name']} — {fname[:40]}")

    st.divider()

    # ── Active tools summary ──
    st.markdown("**Active Tools**")
    st.markdown("- 🔍 DuckDuckGo")
    st.markdown("- 📄 Arxiv")
    st.markdown("- 📖 Wikipedia")
    for url, info in st.session_state.url_sources.items():
        st.markdown(f"- 🌐 `{info['tool_name']}` — {info['title'][:30]}")
    for fname, info in st.session_state.pdf_sources.items():
        st.markdown(f"- 📕 `{info['tool_name']}` — {fname[:30]}")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


#  Page header 
st.title("🔎 LangChain Search Agent")
st.caption(
    "Powered by **Groq LLaMA-3.3-70b** · "
    "Tools: DuckDuckGo · Arxiv · Wikipedia · Multiple URLs · Multiple PDFs · "
    "🧠 Conversation memory"
)


#  Render existing messages ─
source_map = _build_source_map()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        steps = msg.get("trace", [])
        if steps:
            _render_trace(steps, source_map=source_map, collapsed=True)


#  Chat input & agent run ─
if prompt := st.chat_input("Ask anything…"):

    if not api_key:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar first.")
        st.stop()

    # Save and show user message BEFORE building history so we don't include
    # the current question in the history block (it's passed separately).
    history_text = build_chat_history(st.session_state.messages, max_turns=memory_turns)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── Show a small "memory pill" when history is active ──
        if history_text:
            turn_count = history_text.count("\nHuman:") + (1 if history_text.startswith("Human:") else 0)
            st.caption(f"🧠 Using last {memory_turns} messages as context for follow-ups")

        try:
            # ── Assemble all tools (custom first for routing priority) 
            custom_tools = (
                [info["tool"] for info in st.session_state.url_sources.values()]
                + [info["tool"] for info in st.session_state.pdf_sources.values()]
            )
            all_tools = custom_tools + build_base_tools()

            # ── Build sources block for the prefix ──
            source_lines = []
            for url, info in st.session_state.url_sources.items():
                source_lines.append(
                    f"  • {info['tool_name']} → Webpage: '{info['title']}' ({url})"
                )
            for fname, info in st.session_state.pdf_sources.items():
                source_lines.append(
                    f"  • {info['tool_name']} → PDF file: '{fname}'"
                )

            if source_lines:
                sources_block = (
                    "The following custom knowledge sources have been loaded:\n"
                    + "\n".join(source_lines)
                    + "\n\nFor questions related to any of these sources, call the "
                    "corresponding tool FIRST. Only fall back to other tools if it "
                    "returns insufficient information.\n\n"
                )
            else:
                sources_block = ""

            # ── Inject conversation history into the agent prefix 
            # This is the KEY addition: the agent sees prior turns as part of
            # its system-level context so pronouns / references are resolved.
            if history_text:
                history_block = (
                    "CONVERSATION HISTORY (most recent exchanges):\n"
                    "\n"
                    f"{history_text}\n"
                    "\n"
                    "Use this history to resolve references like 'them', 'it', 'those', "
                    "'the above topic', etc. in the current question. "
                    "If the current question is a direct follow-up answerable from the "
                    "history alone, answer directly WITHOUT calling any tool.\n\n"
                )
            else:
                history_block = ""

            prefix = (
                "You are a helpful research assistant with access to multiple tools "
                "and the ability to remember recent conversation.\n\n"
                + history_block
                + sources_block
                + "INSTRUCTIONS:\n"
                "1. Read the conversation history (if any) FIRST to understand context.\n"
                "2. Resolve all pronouns/references using that context before acting.\n"
                "3. Think carefully about which tool(s) are most relevant.\n"
                "4. Call multiple tools to cross-reference when needed.\n"
                "5. In your Final Answer, state which tool(s) you used and what each contributed.\n\n"
                "You have access to the following tools:"
            )

            #  Augment the user prompt with history (belt + suspenders) 
            # The history is passed BOTH in the prefix (system level) AND
            # prepended to the human input so it survives the React template.
            augmented_prompt = augment_prompt_with_history(prompt, history_text)

            #  LLM 
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                streaming=False,
                temperature=0,
            )

            #  Agent 
            agent = initialize_agent(
                tools=all_tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                max_iterations=12,
                early_stopping_method="generate",
                return_intermediate_steps=True,
                agent_kwargs={"prefix": prefix},
            )

            #  Live thought stream 
            live_area = st.empty()
            with live_area.container():
                st_cb = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True,
                    max_thought_containers=15,
                )
                response = agent.invoke(
                    {"input": augmented_prompt},
                    callbacks=[st_cb],
                )
            live_area.empty()

            final_answer       = response.get("output", str(response))
            intermediate_steps = response.get("intermediate_steps", [])
            current_source_map = _build_source_map()

            #  Source attribution badges 
            if intermediate_steps:
                used_tools = list(dict.fromkeys(a.tool for a, _ in intermediate_steps))
                badge_parts = []
                for t in used_tools:
                    icon  = _tool_icon(t)
                    label = current_source_map.get(t, t)
                    badge_parts.append(f"{icon} **{t}** _{label}_")
                st.markdown("**Sources used:** " + "  |  ".join(badge_parts))
                st.markdown("")

            #  Structured trace 
            if intermediate_steps:
                _render_trace(intermediate_steps, source_map=current_source_map, collapsed=False)
            else:
                st.info("ℹ️ The agent answered directly from conversation history (no tool call needed).")

            #  Final answer 
            st.markdown("---")
            st.markdown("### ✅ Final Answer")
            st.markdown(final_answer)

            # Store the clean answer (not the augmented prompt) in history
            st.session_state.messages.append({
                "role":    "assistant",
                "content": final_answer,
                "trace":   intermediate_steps,
            })

        except Exception as e:
            err_msg = f"⚠️ Something went wrong: `{e}`"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})