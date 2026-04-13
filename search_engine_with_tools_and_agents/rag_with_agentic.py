import os
import certifi
import tempfile # for making temporary file
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# these are the langchain tools whose work is to take to query  of the user and call ther relevant api and bring the result
from langchain_classic.agents import initialize_agent, AgentType  # this is use to build agent and which can decide which tool to call for getting result of query     
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler #  Agent ke sochne ke process (thoughts) ko real-time Streamlit UI mein dikhane ke liye.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

from dotenv import load_dotenv
load_dotenv()


# Page config
st.set_page_config(
    page_title="LangChain Search Agent",
    page_icon="🔎",
    layout="wide",
)


# Session-state init  (run once before any widget)
# url_sources  : {url:  {"tool": LangChain tool, "title": str, "tool_name": str}}
# pdf_sources  : {name: {"tool": LangChain tool, "tool_name": str}}
for key, default in [
    ("messages", None),
    ("url_sources", {}),
    ("pdf_sources", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default 
        # ye Loop har key ke liye check karega agar pehle se session state mein nahi hai toh default value set kare.

if st.session_state.messages is None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can search the web, look up research papers, Wikipedia, "
                "answer questions from URLs you add, or read PDFs you upload.\n\n"
                "Use the sidebar to add sources, then ask me anything."
            ),
        }
    ]

## above code meaning is when the code start first time the message is empty so it add the first welcome message by the assistant
# role -> message ke user ya assistant ho sakta hai


# Helpers — build individual tools
@st.cache_resource #st.cache_resource is a Streamlit decorator used to cache expensive, non-serializable objects -> Jab app dobara run ho, ye function dubara execute nahi hoga, pehle wala result use karega. Kyuki tools baar-baar banane ki zaroorat nahi.
def build_base_tools():
    arxiv  = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400))
    wiki   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400))
    search = DuckDuckGoSearchRun(name="DuckDuckGo_Search")
    return [search, arxiv, wiki]


# Common function for URL/PDF
def _make_retriever_tool(docs, tool_name: str, description: str):
    """Shared FAISS embedding + tool creation for both URLs and PDFs."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb   = FAISS.from_documents(chunks, embeddings)
    retriever  = vectordb.as_retriever(search_kwargs={"k": 4})

    #create_retriever_tool – Ye retriever ko ek LangChain tool mein wrap kar deta hai. Jab agent ye tool call karega, toh internally retriever relevant chunks dhundhega aur LLM ko dega.
    return create_retriever_tool(retriever, tool_name, description)


# this function is for -> making tool from url
def load_url_tool(url: str, index: int):
    """
    Load a URL, embed it, return (tool, page_title).
    tool_name is URL_Source_N — short enough for the LLM to reference cleanly.
    """
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


# this function is for -> making tool from pdf
def load_pdf_tool(file_bytes: bytes, filename: str, index: int):
    """
    Save PDF to a temp file, load + embed it, return tool.
    tool_name is PDF_Source_N.
    """
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


# _render_trace — MUST be defined before the message-rendering loop
def _tool_icon(name: str) -> str:
    if name == "DuckDuckGo_Search":  return "🔍"
    if name == "arxiv":              return "📄"
    if name == "wikipedia":          return "📖"
    if name.startswith("URL_"):      return "🌐"
    if name.startswith("PDF_"):      return "📕"
    return "🔧"


def _render_trace(steps: list, source_map: dict, collapsed: bool = False):
    """
    Render the agent reasoning trace.
    source_map : {tool_name -> human-readable label}  e.g. {"URL_Source_1": "colah.github.io — Understanding LSTMs"}
    """
    with st.expander(
        f"📋 Agent reasoning — {len(steps)} tool call(s)",
        expanded=not collapsed,
    ):
        #  Compact summary table 
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

        # Full detail per step 
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


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.divider()

    #  URL section 
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
                idx  = len(st.session_state.url_sources) + 1
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

    # Show loaded URLs with remove buttons
    if st.session_state.url_sources:
        st.markdown("**Loaded URLs:**")
        for url, info in list(st.session_state.url_sources.items()):
            c1, c2 = st.columns([4, 1])
            c1.caption(f"🌐 {info['tool_name']} — {info['title'][:40]}")
            if c2.button("✕", key=f"rm_url_{url}"):
                del st.session_state.url_sources[url]
                st.rerun()

    st.divider()

    #  PDF section 
    st.markdown("### 📕 Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process any newly uploaded PDFs not yet in session state
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

    # Show loaded PDFs
    if st.session_state.pdf_sources:
        st.markdown("**Loaded PDFs:**")
        for fname, info in st.session_state.pdf_sources.items():
            st.caption(f"📕 {info['tool_name']} — {fname[:40]}")

    st.divider()

    #  Active tools summary 
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


# Page header
st.title("🔎 LangChain Search Agent")
st.caption(
    "Powered by **Groq LLaMA-3.3-70b** · "
    "Tools: DuckDuckGo · Arxiv · Wikipedia · Multiple URLs · Multiple PDFs"
)


# Build source_map for trace labels  {tool_name -> human label}
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


# Render existing messages
source_map = _build_source_map()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        steps = msg.get("trace", [])
        if steps:
            _render_trace(steps, source_map=source_map, collapsed=True)


# Chat input & agent run
if prompt := st.chat_input("Ask anything…"):

    if not api_key:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            #  Assemble all tools 
            # Order: URL/PDF sources first (higher routing priority), then base tools
            custom_tools = (
                [info["tool"] for info in st.session_state.url_sources.values()]
                + [info["tool"] for info in st.session_state.pdf_sources.values()]
            )
            all_tools = custom_tools + build_base_tools()

            #  Build agent prefix 
            # List every loaded source explicitly so the LLM knows what's available
            # and which tool_name maps to which content.
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
                    "The following custom knowledge sources have been loaded for you:\n"
                    + "\n".join(source_lines)
                    + "\n\nFor questions related to any of these sources, you MUST call "
                    "the corresponding tool FIRST (before Wikipedia or web search). "
                    "Only fall back to other tools if the custom tool returns insufficient information.\n\n"
                )
            else:
                sources_block = ""

            prefix = (
                "You are a research assistant with access to multiple tools.\n\n"
                + sources_block
                + "IMPORTANT INSTRUCTIONS:\n"
                "1. Think carefully about which tool(s) are most relevant for the question.\n"
                "2. You are ENCOURAGED to call more than one tool — use multiple tools to "
                "cross-reference information and give a comprehensive answer.\n"
                "3. After each tool call evaluate whether the observation fully answers "
                "the question. If not, call another tool.\n"
                "4. In your Final Answer you MUST explicitly state which tool(s) you used "
                "(use the exact tool name, e.g. URL_Source_1, PDF_Source_2, wikipedia) "
                "and what each one contributed.\n\n"
                "You have access to the following tools:"
            )

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
                response = agent.invoke({"input": prompt}, callbacks=[st_cb])
            live_area.empty()

            final_answer       = response.get("output", str(response))
            intermediate_steps = response.get("intermediate_steps", [])
            current_source_map = _build_source_map()

            #  Source attribution badge 
            # Show which tools were actually called, highlighted as badges
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
                st.info("ℹ️ The agent answered directly without calling any tools.")

            #  Final answer 
            st.markdown("---")
            st.markdown("### ✅ Final Answer")
            st.markdown(final_answer)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": final_answer,
                "trace":   intermediate_steps,
            })

        except Exception as e:
            err_msg = f"⚠️ Something went wrong: `{e}`"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})