import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
import datetime
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def crawl_kompasiana_links(base_url, max_pages=30):
    all_links = set()
    for page in tqdm(range(1, max_pages+1), desc="Mengumpulkan Link Kompasiana"):
        url = base_url if page == 1 else f"{base_url}?page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                print(f"hentikan di page {page}, status: {res.status_code}")
                break
            soup = BeautifulSoup(res.text, 'html.parser')
            for artikel_div in soup.find_all('div', class_='artikel'):
                konten = artikel_div.find('div', class_='artikel--content')
                if konten:
                    h2 = konten.find('h2')
                    if h2:
                        a = h2.find('a', href=True)
                        if a and a['href'].startswith('https://www.kompasiana.com/'):
                            all_links.add(a['href'])
            time.sleep(1)
        except Exception as e:
            print(f"Error di page {page}: {e}")
            break
    return list(all_links)

def save_links_to_csv(links):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"artikel_links_kompasiana_{ts}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        for link in links:
            writer.writerow([link])
    print(f"{len(links)} link artikel berhasil disimpan ke {fname}")

# Ubah base_url sesuai link kategori/tag utama
base_url = "https://www.kompasiana.com/tag/dunia-kerja"
links = crawl_kompasiana_links(base_url, max_pages=30)
save_links_to_csv(links)
