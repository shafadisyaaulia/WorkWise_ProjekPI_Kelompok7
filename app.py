import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import time
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="WorkWise Search",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    :root {
        --card-bg: linear-gradient(180deg, rgba(255,255,255,0.66), rgba(255,255,255,0.52));
        --glass-warm: linear-gradient(135deg, rgba(255,183,102,0.06), rgba(255,214,102,0.03));
        --glass-cool: linear-gradient(135deg, rgba(6,182,212,0.06), rgba(52,211,153,0.03));
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    html, body {
        background: linear-gradient(135deg, #cffafe 0%, #d1fae5 50%, #bfdbfe 100%);
        background-attachment: fixed;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(-45deg, #cffafe 0%, #d1fae5 25%, #bfdbfe 50%, #cffafe 75%, #d1fae5 100%);
        background-size: 400% 400%;
        background-attachment: fixed;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        position: relative;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    }
    
    .main .block-container {
        padding-top: 80px;
        max-width: 1400px;
        background: transparent;
    }
    
    /* Navigation Bar */
    .navbar {
        background: linear-gradient(90deg, rgba(255,214,102,0.06), rgba(6,182,212,0.06));
        backdrop-filter: blur(6px);
        padding: 18px 40px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin: -2rem -2rem 2rem -2rem;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .navbar-menu {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    
    .nav-link {
        color: #4b5563;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        padding: 8px 16px;
        border-radius: 8px;
    }
    
    .nav-link:hover {
        color: #667eea;
        background: #667eea10;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 50px 20px 40px 20px;
        margin-bottom: 30px;
        background: linear-gradient(135deg, rgba(255,183,102,0.12) 0%, rgba(255,214,102,0.06) 100%);
        border-radius: 18px;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255,183,102,0.08);
        box-shadow: 0 12px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.06);
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #92400e;
        font-size: 1.15rem;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }
    
    .stat-card {
        background: var(--card-bg);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 12px -6px rgba(0, 0, 0, 0.04);
        transition: all 0.28s ease;
        border: 1px solid rgba(229,231,235,0.6);
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.1);
    }
    
    .stat-icon {
        font-size: 2.2rem;
        margin-bottom: 14px;
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Features Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-bottom: 50px;
    }
    
    .feature-card {
        background: var(--card-bg);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 6px 14px -8px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(229,231,235,0.6);
        transition: all 0.28s ease;
    }
    
    .feature-card:hover {
        border-color: #06b6d4;
        box-shadow: 0 12px 24px -4px rgba(6, 182, 212, 0.15);
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 16px;
    }
    
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Search Section */
    .search-section {
        background: linear-gradient(135deg, rgba(6,182,212,0.12) 0%, rgba(52,211,153,0.08) 100%);
        border-radius: 20px;
        padding: 36px 36px;
        margin-bottom: 36px;
        box-shadow: 0 18px 36px -12px rgba(6, 182, 212, 0.10);
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .search-section::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18%;
        animation: float 6s ease-in-out infinite;
        transform: rotate(12deg);
        filter: blur(6px);
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .search-title {
        text-align: center;
        font-size: 2.0rem;
        font-weight: 800;
        color: white;
        margin-bottom: 18px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        position: relative;
        z-index: 1;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.96);
        border: 2px solid rgba(229,231,235,0.6);
        border-radius: 40px;
        padding: 14px 28px;
        color: #1f2937;
        font-size: 1.05rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        transition: all 0.25s ease;
        position: relative;
        z-index: 1;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(255,255,255,0.9);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.18), 0 0 0 5px rgba(255, 255, 255, 0.18);
        transform: translateY(-3px);
        background: rgba(255,255,255,0.96);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
        font-size: 1.05rem;
    }
    
    /* Team Section */
    .team-section {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 50px 40px;
        margin: 50px 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(229,231,235,0.6);
    }
    
    .section-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 16px;
    }
    
    .section-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 45px;
    }
    
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 30px;
        margin-top: 35px;
    }
    
    .team-card {
        background: linear-gradient(135deg, rgba(6,182,212,0.06) 0%, rgba(52,211,153,0.04) 100%);
        border-radius: 18px;
        padding: 18px 18px 22px 18px;
        text-align: center;
        transition: transform 0.36s cubic-bezier(.2,.9,.2,1), box-shadow 0.36s;
        border: 1px solid rgba(229,231,235,0.5);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
    }

    .team-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 18px 40px rgba(6,182,212,0.08);
        border-color: rgba(6,182,212,0.28);
    }

    .team-photo-wrap {
        width: 180px;
        height: 210px;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        border: 5px solid rgba(255,255,255,0.92);
        box-shadow: 0 8px 22px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.28);
        background: linear-gradient(120deg, rgba(255,255,255,0.02), rgba(6,182,212,0.03));
        transform-origin: center;
    }

    .team-photo-wrap img.team-photo {
        width: 100%;
        height: 100%;
        object-fit: contain;
        object-position: center;
        border-radius: 12px;
        display: block;
        transition: transform 0.6s cubic-bezier(.2,.9,.2,1);
    }

    .team-card:hover .team-photo-wrap img.team-photo {
        transform: scale(1.06) translateY(-4px);
    }

    .team-photo-overlay {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 10px 12px;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.45) 100%);
        color: #fff;
        font-weight: 600;
        font-size: 0.95rem;
        opacity: 0;
        transform: translateY(6px);
        transition: opacity 0.28s, transform 0.28s;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .team-card:hover .team-photo-overlay {
        opacity: 1;
        transform: translateY(0);
    }
    
    .team-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    .team-npm {
        color: #06b6d4;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 10px;
        font-family: 'Courier New', monospace;
    }
    
    .team-role {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 15px;
    }
    
    .team-badge {
        display: inline-block;
        background: linear-gradient(135deg, #06b6d4 0%, #34d399 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Results Section */
    .section-header {
        color: #0891b2;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding: 18px 24px;
        background: linear-gradient(135deg,rgba(191,219,254,0.4),rgba(209,250,229,0.4));
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(8,145,178,0.12), 0 2px 6px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(6,182,212,0.2);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .result-card {
        background: linear-gradient(135deg,rgba(255,255,255,0.8),rgba(240,249,255,0.6));
        border: 1px solid rgba(6,182,212,0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
        transition: all 0.28s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .result-card:hover::before {
        opacity: 1;
    }
    
    .result-title {
        color: #06b6d4;
        font-size: 1.15rem;
        font-weight: 700;
        text-decoration: none;
        display: inline-block;
        margin-bottom: 8px;
        transition: all 0.2s ease;
        line-height: 1.4;
    }
    
    .result-title:hover {
        color: #764ba2;
    }
    
    .result-url {
        color: #10b981;
        font-size: 0.85rem;
        margin-bottom: 12px;
        font-weight: 500;
        word-break: break-all;
    }
    
    .result-snippet {
        color: #4b5563;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 14px;
    }
    
    .source-badge {
        background: linear-gradient(135deg, #06b6d4 0%, #34d399 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        background: var(--card-bg);
        padding: 40px;
        margin-top: 60px;
        border-radius: 20px;
        box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.06);
        text-align: center;
        border: 1px solid rgba(229,231,235,0.6);
    }
    
    .footer-content {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 15px;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-top: 20px;
    }
    
    .footer-link {
        color: #06b6d4;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .footer-link:hover {
        color: #764ba2;
    }
    
    .loading-container {
        text-align: center;
        padding: 30px;
        color: #06b6d4;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    .loading-dots {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    .no-results {
        text-align: center;
        color: #9ca3af;
        padding: 40px;
        background: var(--card-bg);
        border-radius: 16px;
        border: 2px dashed rgba(229,231,235,0.6);
    }
    
    .info-box {
        background: var(--card-bg);
        border: 2px solid rgba(229,231,235,0.6);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        color: #4b5563;
        font-weight: 500;
        margin: 20px 0;
        font-size: 1rem;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f3f4f6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #06b6d4 0%, #34d399 100%);
        border-radius: 5px;
    }

    /* CTA button */
    .cta-btn {
        display: inline-block;
        background: linear-gradient(90deg,#0891b2,#06b6d4);
        color: white !important;
        padding: 12px 26px;
        border-radius: 999px;
        font-weight: 700;
        text-decoration: none;
        box-shadow: 0 8px 24px rgba(6,182,212,0.18);
        transition: transform 0.18s, box-shadow 0.18s;
    }

    .cta-btn:hover { transform: translateY(-3px); box-shadow: 0 14px 34px rgba(6,182,212,0.2); }

    /* subtle animated accent */
    .hero-accent {
        display:inline-block;
        transform-origin:center;
        animation: float-accent 4s ease-in-out infinite;
    }

    @keyframes float-accent { 0%,100%{ transform: translateY(0);} 50%{ transform: translateY(-6px);} }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideDownFade {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes floatWave {
        0%, 100% { transform: translateY(0px) scale(1); }
        50% { transform: translateY(-8px) scale(1.02); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 12px 32px rgba(8,145,178,0.2); }
        50% { box-shadow: 0 12px 40px rgba(8,145,178,0.35); }
    }
    
    /* Splash Page Animations */
    @keyframes titleWobble {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        25% { transform: translateY(-3px) rotate(-0.5deg); }
        50% { transform: translateY(-6px) rotate(0.5deg); }
        75% { transform: translateY(-3px) rotate(-0.5deg); }
    }
    
    @keyframes subtitleFade {
        0% { opacity: 0; transform: translateY(10px); letter-spacing: 0.05em; }
        100% { opacity: 1; transform: translateY(0); letter-spacing: 0.02em; }
    }
    
    @keyframes buttonBounce {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-8px) scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes borderGlow {
        0%, 100% { border-color: rgba(6, 182, 212, 0.3); box-shadow: 0 0 10px rgba(6, 182, 212, 0.2); }
        50% { border-color: rgba(6, 182, 212, 0.8); box-shadow: 0 0 20px rgba(6, 182, 212, 0.4); }
    }
    
    .splash-title {
        animation: slideDownFade 0.8s ease-out, titleWobble 4s ease-in-out infinite;
        animation-delay: 0.1s, 1.2s;
        text-shadow: 0 4px 20px rgba(8, 145, 178, 0.15);
        letter-spacing: -0.5px;
    }
    
    .splash-subtitle {
        animation: subtitleFade 1.2s ease-out 0.3s backwards;
        position: relative;
    }
    
    .splash-subtitle::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.6), transparent);
        animation: fadeInUp 1s ease-out 0.5s backwards;
    }
    
    .splash-button {
        animation: fadeInUp 1s ease-out 0.5s backwards, buttonBounce 2.5s ease-in-out infinite, borderGlow 3s ease-in-out infinite;
        position: relative;
        border: 2px solid rgba(6, 182, 212, 0.3);
        overflow: hidden;
    }
    
    .splash-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: shimmer 3s infinite;
    }
    @media (max-width: 1024px) {
        .main-title { font-size: 2.6rem; }
        .search-title { font-size: 1.85rem; }
        .team-photo-wrap { width: 160px; height: 110px; }
        .team-grid { gap: 22px; }
        .stats-container { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
    }

    @media (max-width: 768px) {
        .navbar { padding: 12px 18px; margin: -1.5rem -1.5rem 1.5rem -1.5rem; }
        .hero-section { padding: 28px 18px; }
        .main-title { font-size: 2.2rem; }
        .subtitle { font-size: 0.98rem; }
        .features-grid { grid-template-columns: 1fr; }
        .team-grid { grid-template-columns: 1fr; }
        .stats-container { grid-template-columns: repeat(2,1fr); }
        .team-photo-wrap { width: 100%; height: 140px; border-radius: 12px; }
        .team-card { flex-direction: column; align-items: center; }
        .search-section { padding: 20px; }
        .stTextInput > div > div > input { width: 100% !important; }
    }

    @media (max-width: 420px) {
        .main-title { font-size: 1.9rem; }
        .search-title { font-size: 1.4rem; }
        .team-photo-wrap { height: 110px; }
        .stat-value { font-size: 1.4rem; }
        .feature-icon { font-size: 2rem; }
        .stTextInput > div > div > input { padding: 10px 16px; font-size: 0.95rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD CORPUS ---
@st.cache_resource
def load_data():
    df_all = pd.read_csv("corpus_final_stemmed.csv")
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df_all['body_stemmed'])
    
    tokenized_corpus = [doc.split() for doc in df_all['body_stemmed']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return df_all, tfidf, X, bm25

df_all, tfidf, X, bm25 = load_data()

# --- PAGE ROUTING (query param 'page=search' opens standalone search page) ---
# Use `st.query_params` (replacement for experimental_get_query_params)
params = st.query_params
page_param = params.get("page")
if isinstance(page_param, list):
    page_value = page_param[0] if page_param else ""
else:
    page_value = page_param or ""
is_search_page = page_value == "search"
# Interpret page values:
# - empty string (no param) -> splash (minimal start screen)
# - 'landing' -> full landing/landing + embedded search
# - 'search' -> standalone search page (other sections hidden)
is_landing = page_value == "landing"
is_splash = page_value == ""

# Minimal splash screen: show only project title and Start button
if is_splash:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;padding:80px 20px;'>
        <h1 class='main-title splash-title' style='font-size:4rem;margin-bottom:8px;'>WorkWise</h1>
        <p class='splash-subtitle' style='color:#6b7280;font-size:1.05rem;margin-bottom:24px;'>Project WorkWise &#8212; Advanced Information Retrieval</p>
        <div style='margin-top:28px;'>
            <a href='?page=landing' class='cta-btn splash-button' style='font-size:1.05rem;padding:14px 28px;'>Start</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- SEARCH FUNCTIONS ---
def simple_search(query, tfidf, X, df, n=10):
    q_vec = tfidf.transform([query])
    scores = (X * q_vec.T).toarray()
    top_idx = scores.flatten().argsort()[::-1][:n]
    return df.iloc[top_idx][['title', 'body_stemmed', 'sumber', 'url']]

def bm25_search(query, bm25, df, n=10):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return df.iloc[top_idx][['title', 'body_stemmed', 'sumber', 'url']]

def render_results(results, model_name, icon):
    st.markdown(f"<div class='section-header'>{icon} {model_name}</div>", unsafe_allow_html=True)
    
    if len(results) == 0:
        st.markdown("<div class='no-results'>&#128237; No results found</div>", unsafe_allow_html=True)
        return
    
    for idx, row in results.iterrows():
        snippet = row['body_stemmed'][:200] + '...' if len(row['body_stemmed']) > 200 else row['body_stemmed']
        
        st.markdown(f"""
        <div class='result-card'>
            <a href="{row['url']}" target="_blank" class='result-title'>
                {row['title']}
            </a>
            <div class='result-url'>&#128279; {row['url']}</div>
            <div class='result-snippet'>{snippet}</div>
            <span class='source-badge'>{row['sumber']}</span>
        </div>
        """, unsafe_allow_html=True)

# --- NAVIGATION BAR ---
if not is_search_page:
    st.markdown("""
<div class='navbar'>
    <div class='navbar-brand'>
        &#128269; WorkWise
    </div>
    <div class='navbar-menu'>
        <a href='?page=landing' class='nav-link'>Home</a>
        <a href='?page=search' class='nav-link'>Search</a>
        <a href='#about' class='nav-link'>About Us</a>
        <a href='#team' class='nav-link'>Our Team</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
if not is_search_page:
    st.markdown("""
    <div class='hero-section' id='home'>
    <h1 class='main-title'>WorkWise Search</h1>
    <p class='subtitle'>&#128640; Advanced AI-Powered Information Retrieval System</p>
    <div style='margin-top:10px;display:flex;flex-direction:column;align-items:center;gap:18px;'>
        <a href='?page=search' class='cta-btn' style='margin-top:6px;'>Start Searching</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- STATS DASHBOARD ---
total_docs = len(df_all)
unique_sources = df_all['sumber'].nunique()

if not is_search_page:
    st.markdown(f"""
<div class='stats-container'>
    <div class='stat-card'>
        <div class='stat-icon'>&#128218;</div>
        <div class='stat-value'>{total_docs:,}</div>
        <div class='stat-label'>Total Documents</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>&#127760;</div>
        <div class='stat-value'>{unique_sources}</div>
        <div class='stat-label'>Data Sources</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>&#9889;</div>
        <div class='stat-value'>2</div>
        <div class='stat-label'>Search Algorithms</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>&#127919;</div>
        <div class='stat-value'>99%</div>
        <div class='stat-label'>Accuracy Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FEATURES GRID ---
if not is_search_page:
    st.markdown("""
<div class='features-grid'>
    <div class='feature-card'>
        <div class='feature-icon'>&#128269;</div>
        <div class='feature-title'>Smart Search</div>
        <div class='feature-desc'>Advanced TF-IDF and BM25 algorithms for precise document retrieval</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>&#9889;</div>
        <div class='feature-title'>Lightning Fast</div>
        <div class='feature-desc'>Optimized search performance with instant results</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>&#128202;</div>
        <div class='feature-title'>Dual Ranking</div>
        <div class='feature-desc'>Compare results from two powerful ranking methods side-by-side</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>&#127912;</div>
        <div class='feature-title'>Clean Interface</div>
        <div class='feature-desc'>Modern, intuitive design for seamless user experience</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SEARCH SECTION (standalone search page) ---
if is_search_page:
    st.markdown("""
<div class='search-section' id='search' style='max-width:1200px;margin:40px auto;padding:48px 36px;width:calc(100% - 80px);background:linear-gradient(135deg,rgba(191,219,254,0.36) 0%,rgba(209,250,229,0.18) 100%);border-radius:20px;box-shadow:0 18px 36px -12px rgba(6, 182, 212, 0.10);backdrop-filter:blur(6px);border:1px solid rgba(6,182,212,0.08);'>
    <div style='display:flex;flex-direction:column;align-items:center;max-width:100%;margin:0 auto;'>
        <h2 class='search-title' style='margin-bottom:12px;font-size:2.4rem;color:#0891b2;'>&#128270; Start Your Search</h2>
        <p style='color:#475569;font-size:1.02rem;margin-bottom:18px;text-align:center;max-width:860px;'>Temukan informasi yang Anda butuhkan dengan dual-algorithm search yang akurat dan cepat</p>
        <div style='width:100%;max-width:1100px;margin:8px auto 6px auto;'>
            <style>
                .stTextInput > div > div > input {
                    width:100% !important;
                    background: rgba(255,255,255,0.96);
                    border: 2px solid rgba(229,231,235,0.6);
                    border-radius: 999px;
                    padding: 16px 34px;
                    color: #1f2937;
                    font-size: 1.05rem;
                    box-shadow: 0 10px 28px rgba(0,0,0,0.08);
                    transition: all 0.25s ease;
                }
                .stTextInput > div > div > input:focus {
                    border-color: rgba(8,145,178,0.95);
                    box-shadow: 0 18px 46px rgba(8,145,178,0.12), 0 0 0 8px rgba(8,145,178,0.06);
                    transform: translateY(-2px);
                }
                .stTextInput > div > div > input::placeholder { color: #9ca3af; font-size:1.03rem; }
            </style>
            <div class='search-input-wrapper' style='position:relative;'>
                <!-- Streamlit input will be rendered below -->
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Ensure the Streamlit text input visually aligns inside the search hero
    st.markdown("""
    <style>
        /* target the widget container that immediately follows the search-section */
        .search-section + div .stTextInput {
            max-width: 1100px;
            width: 90%;
            margin: 6px auto 0 auto;
        }
        .search-section + div .stTextInput > div > div > input {
            width: 100% !important;
            border-radius: 999px;
            padding: 16px 34px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.08);
        }
    </style>
    """, unsafe_allow_html=True)

    # Render an actual Streamlit text input so the standalone search page is interactive
    query = st.text_input(
        label="Search",
        placeholder="Type your keywords here and press Enter...",
        label_visibility="collapsed",
        key="search_input_search"
    )

# --- RESULTS SECTION for standalone search page ---:
if 'query' not in locals():
    query = None

if query:
    st.markdown("<div class='loading-container'><span class='loading-dots'>&#9889; Searching for best results...</span></div>", unsafe_allow_html=True)
    time.sleep(0.3)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        tfidf_res = simple_search(query, tfidf, X, df_all)
        render_results(tfidf_res, "TF-IDF Results", "&#128202;")
    with col2:
        bm25_res = bm25_search(query, bm25, df_all)
        render_results(bm25_res, "BM25 Results", "&#127919;")

if not is_search_page:
    # Initialize session state for button
    if 'run_compare' not in st.session_state:
        st.session_state['run_compare'] = False
    
    # Create the gradient box - centered with text in the middle
    st.markdown("""
    <div style='margin-top: 80px; display: flex; justify-content: center;'>
        <div style='background: linear-gradient(135deg, #06b6d4, #34d399); padding: 60px 80px; border-radius: 20px; color: white; text-align: center; max-width: 1150px; width: 100%;'>
            <div style='font-size: 3rem; font-weight: 800; margin-bottom: 25px;'>Algorithm Comparison</div>
            <p style='font-size: 1.1rem; font-weight: 400; margin: 0; opacity: 0.95; line-height: 1.7;'>Visual comparison of TF-IDF vs BM25 (Overlap@k and rank<br>differences). Click the button to generate or refresh plots.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Place the button centered below the box
    st.markdown("<div style='margin-top: 30px; display: flex; justify-content: center;'>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1.2, 1.6, 1.2])
    with col_c:
        # Custom styling for the button - DARK BLUE
        st.markdown("""
        <style>
            div[data-testid="column"] button[kind="secondary"] {
                background-color: #1e3a8a !important;
                color: white !important;
                padding: 16px 40px !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                font-size: 1.05rem !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3) !important;
                transition: all 0.3s ease !important;
                width: 100% !important;
                white-space: nowrap !important;
            }
            
            div[data-testid="column"] button[kind="secondary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 25px rgba(30, 58, 138, 0.4) !important;
                background-color: #1e40af !important;
            }
            
            div[data-testid="column"] button[kind="secondary"]:active {
                transform: translateY(0) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button('Show Algorithm Comparison Plots', key='show_compare', type='secondary'):
            st.session_state['run_compare'] = True
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Determine whether to run comparison: either session flag or query param
    run_compare = bool(st.session_state.get('run_compare')) or bool(st.query_params.get('compare'))

    try:
        from plot_comparison import find_and_compare
        import os
        
        if run_compare:
            with st.spinner('Generating comparison plots (this may take a few seconds)...'):
                comp_res = find_and_compare(results_dir='.', out_dir='plots')
            
            # Display combined overlap plot
            combined = comp_res.get('combined_plot')
            if combined and os.path.exists(combined):
                st.image(combined, caption='Overlap@k — all topics', use_column_width=True)
            
            # Display per-query metrics table (Precision/Recall/F1) if available
            metrics_csv = os.path.join('plots', 'metrics_per_query.csv')
            if os.path.exists(metrics_csv):
                try:
                    import pandas as pd
                    df_metrics = pd.read_csv(metrics_csv)
                    st.subheader('Per-query Metrics (Precision / Recall / F1)')
                    st.dataframe(df_metrics)
                    
                    # Show aggregate averages per model as a bar chart
                    if {'model', 'precision', 'recall', 'f1'}.issubset(df_metrics.columns):
                        agg = df_metrics.groupby('model')[['precision', 'recall', 'f1']].mean().reset_index()
                        agg = agg.set_index('model')
                        st.subheader('Average Metrics by Model')
                        st.bar_chart(agg)
                except Exception as e:
                    st.warning(f'Could not read metrics CSV: {e}')
            
            # Display confusion matrices (per-topic, two-per-row: BM25 left, TF-IDF right)
            plots_dir = 'plots'
            if os.path.exists(plots_dir):
                conf_files = [f for f in os.listdir(plots_dir) if f.startswith('confusion_') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if conf_files:
                    st.subheader('Confusion Matrices (per-topic)')
                    
                    mapping = {}
                    for f in conf_files:
                        name = os.path.splitext(f)[0]
                        rest = name[len('confusion_'):]
                        if '_' not in rest:
                            continue
                        topic_part, model_part = rest.rsplit('_', 1)
                        mlow = model_part.lower()
                        if 'tf' in mlow or 'tf-idf' in mlow:
                            model_key = 'TF-IDF'
                        else:
                            model_key = 'BM25'
                        mapping.setdefault(topic_part, {})[model_key] = os.path.join(plots_dir, f)
                    
                    for topic in sorted(mapping.keys()):
                        models = mapping[topic]
                        label = topic.replace('_', ' ')
                        
                        outer_title = st.columns([1, 3, 1])
                        with outer_title[1]:
                            st.markdown(f"<div style='text-align:center;margin:8px 0 12px 0;font-weight:700;font-size:1.05rem'>{label}</div>", unsafe_allow_html=True)
                        
                        outer = st.columns([1, 4, 1])
                        with outer[1]:
                            left_col, spacer, right_col = st.columns([1, 0.2, 1])
                            with left_col:
                                p = models.get('BM25')
                                if p and os.path.exists(p):
                                    st.image(p, caption=f"BM25 — {label}", width=320)
                            with right_col:
                                p = models.get('TF-IDF')
                                if p and os.path.exists(p):
                                    st.image(p, caption=f"TF-IDF — {label}", width=320)
    
    except Exception as e:
        st.warning(f'Could not load comparison module or generate plots: {e}')

if is_search_page:
    st.markdown("""
    <div style='max-width:480px;margin:24px auto 0;display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:40px;'>
        <div style='background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(52,211,153,0.06));border-radius:16px;padding:20px;border:1px solid rgba(6,182,212,0.15);box-shadow:0 4px 12px rgba(0,0,0,0.06);'>
            <div style='font-size:1.6rem;margin-bottom:10px;'>&#128161;</div>
            <div style='font-weight:700;color:#0891b2;margin-bottom:8px;font-size:0.95rem;'>Search Tips</div>
            <div style='color:#64748b;font-size:0.85rem;line-height:1.5;'>Enter specific keywords or phrases to find relevant documents. Use quotation marks for exact matches.</div>
        </div>
        <div style='background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(6,182,212,0.06));border-radius:16px;padding:20px;border:1px solid rgba(52,211,153,0.15);box-shadow:0 4px 12px rgba(0,0,0,0.06);'>
            <div style='font-size:1.6rem;margin-bottom:10px;'>⚡</div>
            <div style='font-weight:700;color:#059669;margin-bottom:8px;font-size:0.95rem;'>Dual Algorithm</div>
            <div style='color:#64748b;font-size:0.85rem;line-height:1.5;'>We compare TF-IDF and BM25 algorithms to give you the most accurate and relevant search results.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- ABOUT US SECTION ---
if not is_search_page:
    st.markdown("""
<div class='team-section' id='about'>
    <h2 class='section-title'>&#128214; About WorkWise</h2>
    <p class='section-subtitle'>
        WorkWise is an advanced Information Retrieval system designed to help you find 
        relevant documents quickly and accurately using state-of-the-art search algorithms.
    </p>
</div>
""", unsafe_allow_html=True)

# --- TEAM SECTION ---
if not is_search_page:
    st.markdown("""
    <div class='team-section' id='team'>
        <h2 class='section-title'>&#128101; Meet Our Team Amazing</h2>
        <p class='section-subtitle'>The brilliant minds behind WorkWise Search System</p>
    </div>
    """, unsafe_allow_html=True)

    # --- FUNCTION TEAM CARD ---
    def team_card(img_url, name, npm, role, badge):
        st.markdown(f"""
        <div class='team-card' style='margin-bottom: 10px;'>
            <div class='team-photo-wrap'>
                <img src="{img_url}" class='team-photo' alt='{name}' />
                <div class='team-photo-overlay'>{role}</div>
            </div>
            <div class='team-name' style='margin-top:6px;'>{name}</div>
            <div class='team-npm'>NPM : {npm}</div>
            <div class='team-role' style='margin-bottom:6px;'>{role}</div>
            <span class='team-badge'>{badge}</span>
        </div>
        """, unsafe_allow_html=True)

    img1_url = "https://i.ibb.co.com/35d3Q44Q/team1.jpg"
    img2_url = "https://i.ibb.co.com/TD63H9pF/team2.jpg"
    img3_url = "https://i.ibb.co.com/4njfTPcw/team3.jpg"

    col1, col2, col3 = st.columns(3)
    with col1:
        team_card(img1_url, "Shafa Disya Aulia", "2308107010002", "Project Lead & Backend Developer", "Developer")
    with col2:
        team_card(img2_url, "Bunga Rasikhah Haya", "2308107010010", "Algorithm Specialist", "Developer")
    with col3:
        team_card(img3_url, "Dian Nazira", "2308107010011", "UI/UX Designer & Frontend Developer", "Designer")

# --- FOOTER ---
if not is_search_page:
    st.markdown("""
    <div class='footer'>
        <div class='footer-content'>
            <strong>WorkWise Search</strong> - Advanced Information Retrieval System<br>
            Built with using Streamlit, TF-IDF & BM25 Algorithms<br>
            &copy; 2024 All Rights Reserved
        </div>
        <div class='footer-links'>
            <a href='#' class='footer-link'>Documentation</a>
            <a href='#' class='footer-link'>GitHub</a>
            <a href='#' class='footer-link'>Contact Us</a>
        </div>
    </div>
    """, unsafe_allow_html=True)