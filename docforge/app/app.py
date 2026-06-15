import streamlit as st
from pyfiglet import Figlet


st.set_page_config(
    page_title="DocForge",
    layout="centered",
)

fig = Figlet(font="big")

ascii_art = fig.renderText("DocForge")

st.markdown(
    """
<style>

.stApp {
    background-color: #0b1120;
}

.wrapper {

    margin-top: 80px;

    padding: 50px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #111827,
            #0f172a
        );

    border: 1px solid #1e293b;

    box-shadow:
        0 0 40px rgba(0,255,200,0.08);
}

.ascii {

    overflow-x: auto;

    text-align: center;
}

.ascii pre {

    display: inline-block;

    margin: 0;

    color: #5eead4;

    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 14px;

    line-height: 1.05;

    white-space: pre;

    text-shadow:
        0 0 12px rgba(94,234,212,0.7);
}

.title {

    margin-top: 40px;

    text-align: center;

    color: white;

    font-size: 34px;

    font-weight: 700;
}

.subtitle {

    margin-top: 12px;

    text-align: center;

    color: #94a3b8;

    font-size: 16px;
}

.badge {

    margin-top: 30px;

    text-align: center;
}

.badge span {

    display: inline-block;

    padding: 12px 24px;

    border-radius: 999px;

    border: 1px solid rgba(94,234,212,0.3);

    background: rgba(94,234,212,0.08);

    color: #5eead4;

    font-family: monospace;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="wrapper">

<div class="ascii">
<pre>{ascii_art}</pre>
</div>

<div class="title">
🚧 TO BE IMPLEMENTED 🚧
</div>

<div class="subtitle">
Agentic RAG Document Generation Platform
</div>

<div class="badge">
<span>STATUS :: UNDER DEVELOPMENT</span>
</div>

</div>
""",
    unsafe_allow_html=True,
)