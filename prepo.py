import pandas as pd
import re

df = pd.read_csv("corpus_isi_talenta_FIX.csv")

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # hapus HTML tags
    text = re.sub(r'http\S+', '', text) # hapus URLs
    text = re.sub(r'\s+', ' ', text) # hapus spasi berlebih
    text = text.lower() # lowercase
    # hapus kata promosi umum
    promo_patterns = [
        r'coba gratis talenta', r'jadwalkan demo', r'hubungi sales', r'hubungi kami', r'\bklik\b', r'\bbaca juga\b'
    ]
    for pat in promo_patterns:
        text = re.sub(pat, '', text)
    return text.strip()

df['body_clean'] = df['body'].apply(clean_text)
df.to_csv("corpus_dokumen_final_clean.csv", index=False)
