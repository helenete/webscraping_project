from googlesearch import search
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import requests
import time
import random


class ScrapingUtils:
    def __init__(self) -> None:
        pass

    def get_with_api(url, format, res_page):
        """Fetch all data from an API"""
        params = {
            "format": format,
            "per_page": res_page,
        }

        all_data = []
        page = 1

        while True:
            print(f"Fetching page {page}...")
            params["page"] = page
            response = requests.get(url, params=params)
            data = response.json()
            
            if len(data) < 2 or not data[1]:
                break
            
            all_data.extend(data[1])
            page += 1

        print(f"Fetched {len(all_data)} records.")
        return all_data
        
    def __scrape_with_selenium(url):
        """Scrape an article with Selenium"""
        driver = None
        try:
            # Configuration of Chrome headless
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Wait page loading
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Manage cookies
            cookie_button_selectors = [
                "button[id*='cookie']",
                "button[class*='cookie']",
                "a[id*='cookie']",
                "a[class*='cookie']",
                "button[id*='accept']",
                "button[class*='accept']",
                "#acceptCookies",
                ".accept-cookies",
                "[aria-label*='cookie']",
                "[data-cookiebanner='accept']"
            ]
            
            for selector in cookie_button_selectors:
                try:
                    cookie_button = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    cookie_button.click()
                    break
                except:
                    continue
            
            # Content extraction
            article_data = {
                'url': url,
                'titre': None,
                'contenu': None,
                'annee': None,
                'source': url.split('/')[2]
            }
            
            # Title extraction
            title_selectors = ['h1', '.article-title', '.entry-title', '.post-title']
            for selector in title_selectors:
                try:
                    title = driver.find_element(By.CSS_SELECTOR, selector)
                    article_data['titre'] = title.text
                    break
                except:
                    continue
                    
            if not article_data['titre']:
                article_data['titre'] = driver.title
                
            content_selectors = [
                'article',
                '.article-content',
                '.entry-content',
                'main',
                '#content',
                '.post-content'
            ]
            
            for selector in content_selectors:
                try:
                    content = driver.find_element(By.CSS_SELECTOR, selector)
                    article_data['contenu'] = content.text
                    break
                except:
                    continue
                    
            if not article_data['contenu']:
                # Fallback : get all body
                article_data['contenu'] = driver.find_element(By.TAG_NAME, 'body').text
                
            return article_data
            
        except Exception as e:
            print(f"Erreur lors du scraping de {url}: {str(e)}")
            return None
            
        finally:
            if driver:
                driver.quit()

    def get_and_scrape_articles(self, query, num_results=10):
        """Search and scrape articles"""
        articles_data = []
        successful_scrapes = 0
        
        try:
            urls = list(search(query, lang='en', num_results=num_results))
            total_urls = len(urls)
            
            for i, url in enumerate(urls, 1):
                print(f"Scraping {i}/{total_urls} : {url}")
                
                article_data = self.__scrape_with_selenium(url)
                if article_data and article_data['contenu']:
                    articles_data.append(article_data)
                    successful_scrapes += 1
                    
                # Random delay between requests
                time.sleep(random.uniform(2, 5))
                
            print(f"\nScraping completed : {successful_scrapes}/{total_urls} recovered articles.")
            
            df = pd.DataFrame(articles_data)
            df = df.dropna(subset=['contenu'])
            df = df.drop_duplicates(subset=['titre', 'contenu'])
            
            return df
            
        except Exception as e:
            print(f"Searching error : {str(e)}")
            return pd.DataFrame()