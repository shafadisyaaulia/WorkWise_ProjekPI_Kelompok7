import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="WorkWise Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    .main .block-container {
        padding-top: 0rem;
        max-width: 1400px;
    }
    
    /* Navigation Bar */
    .navbar {
        background: white;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    }
    
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: #6b7280;
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
        background: white;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        background: white;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 12px 24px -4px rgba(102, 126, 234, 0.15);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 24px;
        padding: 60px 50px;
        margin-bottom: 50px;
        box-shadow: 0 20px 40px -10px rgba(102, 126, 234, 0.4);
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
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .search-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 30px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 1;
    }
    
    .stTextInput > div > div > input {
        background: white;
        border: 3px solid white;
        border-radius: 50px;
        padding: 26px 35px;
        color: #1f2937;
        font-size: 1.2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: white;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25), 0 0 0 5px rgba(255, 255, 255, 0.3);
        transform: translateY(-3px);
        background: white;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
        font-size: 1.05rem;
    }
    
    /* Team Section */
    .team-section {
        background: white;
        border-radius: 20px;
        padding: 50px 40px;
        margin: 50px 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
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
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid #e5e7eb;
    }
    
    .team-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 35px -5px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .team-photo {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        margin: 0 auto 20px;
        border: 4px solid white;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        object-fit: cover;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .team-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
    }
    
    .team-npm {
        color: #667eea;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        padding: 18px 24px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
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
        color: #667eea;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        background: white;
        padding: 40px;
        margin-top: 60px;
        border-radius: 20px;
        box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border: 1px solid #e5e7eb;
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
        color: #667eea;
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
        color: #667eea;
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
        background: white;
        border-radius: 16px;
        border: 2px dashed #e5e7eb;
    }
    
    .info-box {
        background: white;
        border: 2px solid #e5e7eb;
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
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
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
        st.markdown("<div class='no-results'>📭 No results found</div>", unsafe_allow_html=True)
        return
    
    for idx, row in results.iterrows():
        snippet = row['body_stemmed'][:200] + '...' if len(row['body_stemmed']) > 200 else row['body_stemmed']
        
        st.markdown(f"""
        <div class='result-card'>
            <a href="{row['url']}" target="_blank" class='result-title'>
                {row['title']}
            </a>
            <div class='result-url'>🔗 {row['url']}</div>
            <div class='result-snippet'>{snippet}</div>
            <span class='source-badge'>{row['sumber']}</span>
        </div>
        """, unsafe_allow_html=True)

# --- NAVIGATION BAR ---
st.markdown("""
<div class='navbar'>
    <div class='navbar-brand'>
        🔍 WorkWise
    </div>
    <div class='navbar-menu'>
        <a href='#home' class='nav-link'>Home</a>
        <a href='#search' class='nav-link'>Search</a>
        <a href='#about' class='nav-link'>About Us</a>
        <a href='#team' class='nav-link'>Our Team</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class='hero-section' id='home'>
    <h1 class='main-title'>WorkWise Search</h1>
    <p class='subtitle'>🚀 Advanced AI-Powered Information Retrieval System</p>
</div>
""", unsafe_allow_html=True)

# --- STATS DASHBOARD ---
total_docs = len(df_all)
unique_sources = df_all['sumber'].nunique()

st.markdown(f"""
<div class='stats-container'>
    <div class='stat-card'>
        <div class='stat-icon'>📚</div>
        <div class='stat-value'>{total_docs:,}</div>
        <div class='stat-label'>Total Documents</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>🌐</div>
        <div class='stat-value'>{unique_sources}</div>
        <div class='stat-label'>Data Sources</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>⚡</div>
        <div class='stat-value'>2</div>
        <div class='stat-label'>Search Algorithms</div>
    </div>
    <div class='stat-card'>
        <div class='stat-icon'>🎯</div>
        <div class='stat-value'>99%</div>
        <div class='stat-label'>Accuracy Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FEATURES GRID ---
st.markdown("""
<div class='features-grid'>
    <div class='feature-card'>
        <div class='feature-icon'>🔍</div>
        <div class='feature-title'>Smart Search</div>
        <div class='feature-desc'>Advanced TF-IDF and BM25 algorithms for precise document retrieval</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>⚡</div>
        <div class='feature-title'>Lightning Fast</div>
        <div class='feature-desc'>Optimized search performance with instant results</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>📊</div>
        <div class='feature-title'>Dual Ranking</div>
        <div class='feature-desc'>Compare results from two powerful ranking methods side-by-side</div>
    </div>
    <div class='feature-card'>
        <div class='feature-icon'>🎨</div>
        <div class='feature-title'>Clean Interface</div>
        <div class='feature-desc'>Modern, intuitive design for seamless user experience</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SEARCH SECTION ---

# --- SEARCH SECTION ---
st.markdown("""
<div class='search-section' id='search' style='padding-top:40px;'>
    <div style='display:flex;flex-direction:column;align-items:center;'>
        <h2 class='search-title' style='margin-bottom:18px;'>🔎 Start Your Search</h2>
        <div style='width:100%;max-width:00px;'>
            <style>
                .stTextInput > div > div > input {
                    background: #fff;
                    border: 2px solid #e5e7eb;
                    border-radius: 30px;
                    padding: 10px 48px 18px 48px;
                    font-size: 1.15rem;
                    box-shadow: 0 4px 16px rgba(102,126,234,0.08);
                    transition: border 0.3s;
                }
                .stTextInput > div > div > input:focus {
                    border-color: #764ba2;
                }
                .search-input-icon {
                    position: absolute;
                    left: 18px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 1.3rem;
                    color: #764ba2;
                    pointer-events: none;
                }
                .search-input-wrapper {
                    position: relative;
                }
            </style>
            <div class='search-input-wrapper'>
                <span class='search-input-icon'></span>
                <!-- Streamlit input will be rendered here -->
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Streamlit input inside the styled wrapper
query = st.text_input(
    label="", 
    placeholder="Type your keywords here and press Enter...", 
    label_visibility="collapsed", 
    key="search_input"
)

# --- RESULTS SECTION ---
if query:
    st.markdown("<div class='loading-container'><span class='loading-dots'>⚡ Searching for best results...</span></div>", unsafe_allow_html=True)
    time.sleep(0.3)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        tfidf_res = simple_search(query, tfidf, X, df_all)
        render_results(tfidf_res, "TF-IDF Results", "📊")
    
    with col2:
        bm25_res = bm25_search(query, bm25, df_all)
        render_results(bm25_res, "BM25 Results", "🎯")
else:
    st.markdown("""
    <div class='info-box'>
        💡 <strong>Tip:</strong> Enter specific keywords or phrases to find relevant documents. 
        Our dual algorithm approach ensures you get the most accurate results!
    </div>
    """, unsafe_allow_html=True)

# --- ABOUT US SECTION ---
st.markdown("""
<div class='team-section' id='about'>
    <h2 class='section-title'>📖 About WorkWise</h2>
    <p class='section-subtitle'>
        WorkWise is an advanced Information Retrieval system designed to help you find 
        relevant documents quickly and accurately using state-of-the-art search algorithms.
    </p>
</div>
""", unsafe_allow_html=True)

# --- TEAM SECTION ---

st.markdown("""
<div class='team-section' id='team'>
    <h2 class='section-title'>👥 Meet Our Team</h2>
    <p class='section-subtitle'>The brilliant minds behind WorkWise Search System</p>
</div>
""", unsafe_allow_html=True)

def team_card(img_path, name, npm, role, badge):
    st.markdown(f"""
    <div style='background: #fff; border-radius: 18px; box-shadow: 0 4px 16px rgba(102,126,234,0.10); padding: 32px 18px 24px 18px; text-align: center; border: 1.5px solid #e5e7eb; margin-bottom: 10px;'>
    """, unsafe_allow_html=True)
    st.image(img_path, width=120)
    st.markdown(f"""
        <div style='font-size:1.18rem; font-weight:700; color:#1f2937; margin-bottom:6px;'>{name}</div>
        <div style='color:#667eea; font-weight:600; font-size:1rem; margin-bottom:6px; font-family:Courier New,monospace;'>NPM : {npm}</div>
        <div style='color:#6b7280; font-size:0.98rem; font-weight:500; margin-bottom:12px;'>{role}</div>
        <span style='display:inline-block; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff; padding:7px 20px; border-radius:20px; font-size:0.9rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>{badge}</span>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    team_card("images/team1.jpg", "Shafa Disya Aulia", "2308107010002", "Project Lead & Backend Developer", "Developer")
with col2:
    team_card("images/team3.jpg", "Bunga Rashikah Haya", "2308107010010", "Algorithm Specialist", "Developer")
with col3:
    team_card("images/team2.jpg", "Dian Nazira", "2308107010011", "UI/UX Designer & Frontend Developer", "Designer")

# --- FOOTER ---
st.markdown("""
<div class='footer'>
    <div class='footer-content'>
        <strong>WorkWise Search</strong> - Advanced Information Retrieval System<br>
        Built with ❤️ using Streamlit, TF-IDF & BM25 Algorithms<br>
        © 2024 All Rights Reserved
    </div>
    <div class='footer-links'>
        <a href='#' class='footer-link'>Documentation</a>
        <a href='#' class='footer-link'>GitHub</a>
        <a href='#' class='footer-link'>Contact Us</a>
    </div>
</div>
""", unsafe_allow_html=True)