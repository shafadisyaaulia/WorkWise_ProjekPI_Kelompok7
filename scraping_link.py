from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import csv
import datetime

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def crawl_kompasiana_links(base_url, max_pages=20):
    all_links = set()
    for page in tqdm(range(1, max_pages+1), desc="Selenium Kompasiana"):
        url = base_url if page == 1 else f"{base_url}?page={page}"
        try:
            driver.get(url)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            # Selector sesuai inspect
            for artikel_div in soup.find_all('div', class_='artikel'):
                konten = artikel_div.find('div', class_='artikel--content')
                if konten:
                    h2 = konten.find('h2')
                    if h2:
                        a = h2.find('a', href=True)
                        if a and a['href'].startswith('https://www.kompasiana.com/'):
                            all_links.add(a['href'])
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

base_url = "https://www.kompasiana.com/tag/dunia-kerja"
links = crawl_kompasiana_links(base_url, max_pages=20)
save_links_to_csv(links)
driver.quit()
