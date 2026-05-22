import streamlit as st
from dotenv import load_dotenv
import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

# =========================
# Load Environment Variables
# =========================

load_dotenv()

# =========================
# Streamlit Config
# =========================

st.set_page_config(
    page_title="City AI Assistant",
    page_icon="🌍",
    layout="centered"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 20px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================

st.markdown('<div class="title">🌍 City AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Get Weather & Latest News Instantly</div>',
    unsafe_allow_html=True
)

# =========================
# Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"🌦️ Weather in {city}: {desc}, {temp}°C"


# =========================
# News Tool
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"""
📰 {title}

🔗 {url}

📝 {snippet[:120]}...
"""
        )

    return "\n".join(news_list)


# =========================
# LLM Setup
# =========================

llm = ChatMistralAI(model="mistral-small-2506")


# =========================
# Human Approval Middleware
# =========================

@wrap_tool_call
def human_approval(request, handler):

    tool_name = request.tool_call["name"]

    approval = st.session_state.get("tool_approval", True)

    if not approval:
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# =========================
# Agent
# =========================

agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant.",
    middleware=[human_approval]
)

# =========================
# Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# Display Chat
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# Chat Input
# =========================

prompt = st.chat_input("Ask about weather or city news...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })

            response = result["messages"][-1].content

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# =========================
# Footer
# =========================

st.markdown(
    '<div class="footer">Built with Streamlit + LangChain + Mistral AI</div>',
    unsafe_allow_html=True
)