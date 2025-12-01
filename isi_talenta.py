from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
import time

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def scrape_articles_from_csv_selenium(csv_file, output_file):
    results = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        urls = [row['url'] for row in reader]
    for url in tqdm(urls, desc="Scraping Isi Talenta (Selenium)"):
        try:
            driver.get(url)
            time.sleep(3)  # Tunggu JS render

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Judul dari <h1>
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else ""
            
            # Konten utama dari <article>
            body = ""
            article_tag = soup.find('article')
            if article_tag:
                paragraphs = article_tag.find_all('p')
                body = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])

            date_tag = soup.find('time')
            date = date_tag.get_text(strip=True) if date_tag else ""

            results.append({"url": url, "title": title, "body": body, "date": date})
            time.sleep(0.5)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'title', 'body', 'date'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Selesai scraping {len(results)} artikel, hasil: {output_file}")
    driver.quit()

if __name__ == "__main__":
    input_csv = "artikel_links_talenta_all_kategori_20251114_224627.csv"  # ganti dengan file link kamu
    output_csv = "corpus_isi_talenta_FIX.csv"
    scrape_articles_from_csv_selenium(input_csv, output_csv)
