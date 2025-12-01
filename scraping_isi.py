import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def scrape_articles_from_csv(csv_file, output_file):
    results = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        urls = [row['url'] for row in reader]
    for url in tqdm(urls, desc="Scraping Isi Kompasiana"):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            title = ""
            body = ""
            date = ""
            # Judul
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)
            # Konten (cek beberapa kemungkinan class)
            content_div = soup.find("div", class_="read__content")
            if not content_div:
                content_div = soup.find("div", class_="content-detail")
            if not content_div:
                content_div = soup.find("div", itemprop="articleBody")
            if not content_div:
                content_div = soup.find("div", class_="post-content")
            if content_div:
                paragraphs = content_div.find_all("p")
                body = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])
            # Tanggal
            date_tag = soup.find("time")
            if date_tag:
                date = date_tag.get_text(strip=True)
            results.append({"url": url, "title": title, "body": body, "date": date})
            time.sleep(1)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'title', 'body', 'date'])
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)} artikel berhasil disimpan ke {output_file}")

# Nama file input & output
csv_file = "artikel_links_kompasiana_20251114_213427.csv"
output_file = "corpus_isi_kompasiana.csv"

scrape_articles_from_csv(csv_file, output_file)
