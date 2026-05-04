from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="MoodBot", page_icon="🎭", layout="centered")

# ── Mode config ───────────────────────────────────────────────────────────────
MODES = {
    "Funny": {
        "system": "You are a funny ai agent",
        "emoji": "😂",
        "label": "Funny AI",
        "accent": "#f5c518",
        "accent_dim": "rgba(245,197,24,0.12)",
        "tag": "Will roast you with love",
    },
    "Angry": {
        "system": "You are an angry ai agent",
        "emoji": "😤",
        "label": "Angry AI",
        "accent": "#ff4444",
        "accent_dim": "rgba(255,68,68,0.12)",
        "tag": "Mad about everything",
    },
    "Sad": {
        "system": "You are a sad ai agent",
        "emoji": "😢",
        "label": "Sad AI",
        "accent": "#5b9cf6",
        "accent_dim": "rgba(91,156,246,0.12)",
        "tag": "Everything hurts a little",
    },
}

# ── Session state ─────────────────────────────────────────────────────────────
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None
if "messages" not in st.session_state:
    st.session_state.messages = []

mode_key = st.session_state.selected_mode
active = MODES.get(mode_key, {})
accent     = active.get("accent", "#f5c518")
accent_dim = active.get("accent_dim", "rgba(245,197,24,0.12)")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');

:root {{
    --bg:      #0a0a0a;
    --surface: #141414;
    --border:  #252525;
    --text:    #eeebe5;
    --muted:   #555;
    --accent:  {accent};
    --accent-dim: {accent_dim};
    --radius:  14px;
    --head:    'Syne', sans-serif;
    --mono:    'DM Mono', monospace;
}}

html, body, [class*="css"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 1.8rem !important;
    padding-bottom: 5rem !important;
    max-width: 800px !important;
}}

/* ── Header ── */
.app-header {{
    text-align: center;
    padding-bottom: 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}}
.app-header h1 {{
    font-family: var(--head) !important;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--accent) !important;
    letter-spacing: -.04em;
    margin: 0 0 .3rem;
}}
.app-header p {{
    font-size: .72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .14em;
    margin: 0;
}}

/* ── Mode cards ── */
.mode-card {{
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 16px 18px;
    text-align: center;
    transition: border-color .2s, transform .15s, box-shadow .2s;
}}
.mode-card.active {{
    border-color: var(--accent);
    background: var(--accent-dim);
    box-shadow: 0 0 0 3px var(--accent-dim);
}}
.mode-card .card-emoji {{ font-size: 2.2rem; display: block; margin-bottom: .5rem; }}
.mode-card .card-label {{
    font-family: var(--head) !important;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
    display: block;
    margin-bottom: .2rem;
}}
.mode-card .card-tag {{
    font-size: .68rem;
    color: var(--muted);
    letter-spacing: .06em;
}}
.mode-card.active .card-label {{ color: var(--accent); }}

/* ── Active mode badge ── */
.mode-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: .78rem;
    color: var(--accent);
    font-family: var(--mono) !important;
    margin-bottom: 1.2rem;
}}

/* ── Chat bubbles ── */
.chat-scroll {{
    max-height: 480px;
    overflow-y: auto;
    padding-right: 4px;
    margin-bottom: 1rem;
}}
.chat-scroll::-webkit-scrollbar {{ width: 4px; }}
.chat-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}

.msg-row {{ display: flex; gap: 10px; margin-bottom: .9rem; align-items: flex-start; }}
.msg-row.user {{ flex-direction: row-reverse; }}

.avatar {{
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    border: 1.5px solid var(--border);
}}
.avatar.user {{ background: var(--accent); border-color: var(--accent); }}
.avatar.ai   {{ background: var(--surface); }}

.bubble {{
    max-width: 70%;
    padding: 11px 15px;
    border-radius: var(--radius);
    font-size: .87rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}}
.bubble.user {{
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-bottom-right-radius: 4px;
}}
.bubble.ai {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-bottom-left-radius: 4px;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 2.5rem 0;
    color: var(--muted);
    font-size: .83rem;
    letter-spacing: .04em;
}}
.empty-state .big {{ font-size: 2.4rem; display: block; margin-bottom: .6rem; }}

/* ── Input ── */
.stTextInput > div > div > input {{
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: .88rem !important;
    padding: 13px 16px !important;
    caret-color: var(--accent);
    transition: border-color .2s;
}}
.stTextInput > div > div > input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}}
.stTextInput > div > div > input::placeholder {{ color: var(--muted) !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: var(--accent) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--head) !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: transform .15s, filter .2s !important;
    letter-spacing: .02em;
}}
.stButton > button:hover {{ filter: brightness(1.12) !important; transform: translateY(-1px) !important; }}

.clear-wrap .stButton > button {{
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    font-family: var(--mono) !important;
    font-size: .74rem !important;
    padding: 5px 18px !important;
    border-radius: 999px !important;
}}
.clear-wrap .stButton > button:hover {{
    border-color: #ff4444 !important;
    color: #ff4444 !important;
    filter: none !important;
    transform: none !important;
}}

hr {{ border-color: var(--border) !important; margin: .4rem 0 1.2rem !important; }}

/* ── Pick hint ── */
.pick-hint {{
    text-align: center;
    padding: 2.5rem 1rem;
    color: var(--muted);
    font-size: .83rem;
    letter-spacing: .06em;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
}}
.pick-hint .big {{ font-size: 2rem; display:block; margin-bottom:.5rem; }}
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return init_chat_model("mistral-small", model_provider="mistralai", temperature=0.9)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🎭 MoodBot</h1>
    <p>pick a mood · start chatting · powered by mistral-small</p>
</div>
""", unsafe_allow_html=True)

# ── Mode selector cards ───────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
for col, m in zip([col1, col2, col3], MODES.keys()):
    cfg = MODES[m]
    is_active = (st.session_state.selected_mode == m)
    active_cls = "active" if is_active else ""
    with col:
        st.markdown(f"""
        <div class="mode-card {active_cls}">
            <span class="card-emoji">{cfg['emoji']}</span>
            <span class="card-label">{cfg['label']}</span>
            <span class="card-tag">{cfg['tag']}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(cfg["label"], key=f"mode_{m}"):
            if st.session_state.selected_mode != m:
                st.session_state.selected_mode = m
                st.session_state.messages = [SystemMessage(content=cfg["system"])]
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── No mode chosen ────────────────────────────────────────────────────────────
if not st.session_state.selected_mode:
    st.markdown("""
    <div class="pick-hint">
        <span class="big">☝️</span>
        Choose an AI mood above to start chatting
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Active badge ──────────────────────────────────────────────────────────────
badge_cfg = MODES[st.session_state.selected_mode]
st.markdown(f"""
<div style="text-align:center; margin-bottom:1rem;">
    <span class="mode-badge">{badge_cfg['emoji']} {badge_cfg['label']} — active</span>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
chat_msgs = [m for m in st.session_state.messages if not isinstance(m, SystemMessage)]

chat_html = '<div class="chat-scroll" id="chatbox">'
if not chat_msgs:
    chat_html += f"""
    <div class="empty-state">
        <span class="big">{badge_cfg['emoji']}</span>
        Say something… I'm in <strong>{badge_cfg['label']}</strong> mode.
    </div>"""
else:
    for msg in chat_msgs:
        if isinstance(msg, HumanMessage):
            chat_html += f"""
            <div class="msg-row user">
                <div class="avatar user">🧑</div>
                <div class="bubble user">{msg.content}</div>
            </div>"""
        elif isinstance(msg, AIMessage):
            chat_html += f"""
            <div class="msg-row ai">
                <div class="avatar ai">{badge_cfg['emoji']}</div>
                <div class="bubble ai">{msg.content}</div>
            </div>"""
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

st.markdown("""
<script>
const c = document.getElementById('chatbox');
if (c) c.scrollTop = c.scrollHeight;
</script>
""", unsafe_allow_html=True)

def handle_send():
    user_text = st.session_state.input.strip()
    if not user_text:
        return

    model = get_model()

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_text))

    # Generate response
    with st.spinner(""):
        response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))

    # Clear input safely
    st.session_state.input = ""

# ── Input ─────────────────────────────────────────────────────────────────────
# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

col_in, col_btn = st.columns([5, 1])

with col_in:
    st.text_input(
        "Message",
        placeholder="Type your message…",
        key="input",
        label_visibility="collapsed",
        on_change=handle_send   # 🔥 FIX: handles Enter key
    )

with col_btn:
    send = st.button("Send →", on_click=handle_send)

