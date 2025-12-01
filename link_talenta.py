from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import csv
import datetime
import re

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Daftar kategori Talenta relevant kategori dunia kerja dan produktivitas
categories = [
    "solusi-talenta",
    "compensation-benefit",
    "organizational-development",
    "news",
    "insight-talenta",
    "benefit-kompensasi",
    "administrasi-hr",
    "performance-management",
    "talent-management",
    "recruitment-selection"
]

def crawl_links_from_category(category, max_pages=10):
    base_url = f"https://www.talenta.co/blog/category/{category}/"
    links = set()
    for page in tqdm(range(1, max_pages+1), desc=f"Crawling {category}"):
        url = base_url if page == 1 else f"{base_url}page/{page}/"
        driver.get(url)
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if (
                href.startswith("https://www.talenta.co/blog/")
                and re.match(r'^https://www\.talenta\.co/blog/[^/]+/$', href)
                and 'category' not in href and 'tag' not in href and 'author' not in href and 'all' not in href
            ):
                links.add(href)
        if len(links) == 0 and page == 1:
            print(f"Tidak ditemukan artikel kategori {category} di halaman {page}")
            break
    return links

if __name__ == "__main__":
    all_links = set()
    for cat in categories:
        cat_links = crawl_links_from_category(cat, max_pages=10)
        all_links.update(cat_links)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"artikel_links_talenta_all_kategori_{ts}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        for link in all_links:
            writer.writerow([link])
    print(f"Total {len(all_links)} link artikel dari {len(categories)} kategori berhasil disimpan ke {filename}")
    driver.quit()
