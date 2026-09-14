import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    /* Hero */
    .hero-banner { background: linear-gradient(135deg, #4f46e5, #7c3aed, #db2777); background-size: 200% 200%; animation: gradientShift 6s ease infinite; border-radius: 20px; padding: 2.2rem 1.5rem 1.8rem; text-align: center; margin-bottom: 1.8rem; box-shadow: 0 10px 40px rgba(79, 70, 229, 0.35); }
    @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .hero-icon { font-size: 2.8rem; animation: bounce 2.5s ease-in-out infinite; display: inline-block; }
    @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
    .hero-title { font-size: 2.2rem; font-weight: 700; color: white; margin: 0.3rem 0 0.2rem; }
    .hero-sub { font-size: 0.95rem; color: rgba(255,255,255,0.8); font-weight: 300; margin: 0; }
    
    /* Status Bar */
    .status-bar { background: linear-gradient(135deg, #ecfdf5, #d1fae5); border: 1px solid #6ee7b7; border-radius: 12px; padding: 0.75rem 1.2rem; display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.5rem; }
    .status-bar span { font-size: 0.85rem; font-weight: 600; color: #065f46; }
    .status-dot { width: 10px; height: 10px; background: #10b981; border-radius: 50%; animation: ping 1.5s infinite; flex-shrink: 0; }
    @keyframes ping { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.4); } }
    
    /* Chat layout */
    .chat-container { border: 1.5px solid #e5e7eb; border-radius: 16px; overflow: hidden; background: #fafafa; margin-bottom: 1rem; }
    .chat-header { background: linear-gradient(135deg, #4f46e5, #7c3aed); padding: 0.85rem 1.2rem; display: flex; align-items: center; gap: 0.5rem; color: white; font-weight: 600; }
    .msg-user { display: flex; justify-content: flex-end; margin-bottom: 0.8rem; }
    .msg-user-bubble { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; border-radius: 18px 18px 4px 18px; padding: 0.7rem 1.1rem; max-width: 80%; font-size: 0.95rem; }
    .msg-bot { display: flex; justify-content: flex-start; margin-bottom: 0.8rem; }
    .msg-bot-avatar { width: 32px; height: 32px; background: linear-gradient(135deg, #4f46e5, #db2777); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; margin-right: 0.5rem; margin-top: 2px; }
    .msg-bot-bubble { background: white; color: #1f2937; border-radius: 18px 18px 18px 4px; padding: 0.7rem 1.1rem; max-width: 80%; font-size: 0.95rem; border: 1px solid #f3f4f6; }
    
    /* Footer & Sidebar */
    .empty-chat { text-align: center; padding: 1.5rem 1rem; color: #9ca3af; }
    .footer { text-align: center; margin-top: 2.5rem; padding: 1.2rem; background: linear-gradient(135deg, #f5f3ff, #fdf2f8); border-radius: 14px; border: 1px solid #e9d5ff; }
    .footer-name { font-size: 1.05rem; font-weight: 700; background: linear-gradient(135deg, #4f46e5, #db2777); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sb-item { background: white; border-radius: 10px; padding: 0.65rem 0.9rem; margin-bottom: 0.5rem; border: 1px solid #f3f4f6; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .sb-label { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; font-weight: 600; }
    .sb-value { font-size: 0.9rem; font-weight: 600; color: #1f2937; margin-top: 1px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-icon">📄</div>
        <h1 class="hero-title">PDF QA Bot</h1>
        <p class="hero-sub">Upload a PDF · Ask anything · Get instant answers</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_bar(filename, page_count, size_kb):
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot"></div>
        <span>✅ "{filename}" ready &nbsp;·&nbsp; {page_count} pages &nbsp;·&nbsp; {size_kb:.1f} KB</span>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(filename, page_count, size_kb):
    with st.sidebar:
        st.markdown("### ⚙️ About this App")
        st.markdown(f"""
        <div class="sb-item"><div class="sb-label">Document</div><div class="sb-value">{filename}</div></div>
        <div class="sb-item"><div class="sb-label">Pages</div><div class="sb-value">{page_count}</div></div>
        <div class="sb-item"><div class="sb-label">File Size</div><div class="sb-value">{size_kb:.1f} KB</div></div>
        <div class="sb-item"><div class="sb-label">AI Model</div><div class="sb-value">GPT-OSS 20B</div></div>
        """, unsafe_allow_html=True)
        st.markdown('<hr style="border:none; border-top:1px solid #f3f4f6; margin:1rem 0;"><div style="text-align:center;"><div style="font-size:0.72rem; color:#9ca3af;">Built by</div><div class="footer-name">Totan</div></div>', unsafe_allow_html=True)

def render_chat_message(role, content):
    if role == "user":
        st.markdown(f'<div class="msg-user"><div class="msg-user-bubble">{content}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot"><div class="msg-bot-avatar">🤖</div><div class="msg-bot-bubble">{content}</div></div>', unsafe_allow_html=True)

def render_footer():
    st.markdown("""
    <div class="footer">
        <div style="font-size:0.8rem; color:#6b7280;">✨ Crafted with ❤️ by</div>
        <div class="footer-name">Totan</div>
        <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 3px;">LangChain · Groq · FAISS · HuggingFace · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)