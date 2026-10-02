import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import tempfile
import os
import uuid
import re
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import agent_executor

st.set_page_config(page_title="OmniQuery Clinical AI", layout="wide", page_icon="🩺")

# --- Custom Clinical UI Theme ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfdfe; }
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        padding: 2rem 1rem;
    }
    
    /* Custom Sidebar Metric Cards to prevent ellipsis truncation */
    .metric-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #e0efff 100%);
        border-left: 4px solid #0369a1;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 5px 0;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        white-space: pre-wrap;
    }

    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to forcefully clean LLM LaTeX output before rendering
def clean_latex(text):
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🩺 OmniQuery Engine")
    st.caption("Agentic Medical RAG Assistant")
    st.divider()
    
    # MOVED: Voice input prioritized at the top
    st.markdown("#### 🎙️ Clinical Voice Input")
    audio = mic_recorder(start_prompt="🎤 Start Voice Query", stop_prompt="🛑 Stop Recording", key="mic")
    
    st.divider()
    
    st.markdown("#### 📊 Architecture Metrics")
    
    # Custom HTML Metric Cards
    st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Vector Store</p>
            <p class="metric-value">FAISS (Local)</p>
        </div>
        <div class="metric-card">
            <p class="metric-title">Embedding Model</p>
            <p class="metric-value">all-MiniLM-L6-v2</p>
        </div>
        <div class="metric-card">
            <p class="metric-title">Active LLM</p>
            <p class="metric-value">Qwen 2.5 (1.5B)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        
    with st.expander("System Diagnostic Logs"):
        st.caption(f"Session Thread: `{st.session_state.thread_id[:8]}`")
        st.caption("FAISS Index: `Active`")
        st.caption("Synthesis: `edge-tts`")

# --- Main Interface ---
st.markdown("<h1 style='text-align: center; color: #0f172a; margin-bottom: 0;'>OmniQuery Clinical UI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>Speak naturally to query complex medical imaging literature.</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_stt_model():
    return WhisperModel("base.en", device="cpu", compute_type="int8")

stt_model = load_stt_model()

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍⚕️"): 
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant", avatar="🩺"): 
            st.markdown(clean_latex(msg.content))

if audio:
    with st.spinner("🎙️ Transcribing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio['bytes'])
            tmp_path = tmp.name
        
        segments, _ = stt_model.transcribe(tmp_path, beam_size=5)
        transcript = "".join([segment.text for segment in segments]).strip()
        os.remove(tmp_path)
    
    if transcript:
        st.session_state.chat_history.append(HumanMessage(content=transcript))
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(transcript)
            
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("🧠 Searching clinical literature and reasoning..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = agent_executor.invoke({"messages": st.session_state.chat_history}, config)
                
                raw_response = result["messages"][-1].content
                agent_response = clean_latex(raw_response)
                
                st.markdown(agent_response)
                st.session_state.chat_history.append(AIMessage(content=raw_response))
                
                for m in result["messages"]:
                    if isinstance(m, ToolMessage):
                        with st.expander("📄 View Retrieved Document Citations"):
                            st.markdown(f"```text\n{m.content}\n```")
                
            with st.spinner("🔊 Synthesizing clinical audio response..."):
                async def synthesize_audio():
                    communicate = edge_tts.Communicate(raw_response, "en-US-AriaNeural")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                        await communicate.save(tmp_audio.name)
                        return tmp_audio.name
                    
                audio_path = asyncio.run(synthesize_audio())
                st.audio(audio_path, format="audio/mp3", autoplay=True)
                os.remove(audio_path)