import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import csv  # Add CSV import

def fetch_page(url, headers=None, max_retries=3):
    """Fetch a page with retries and error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch {url} after {max_retries} attempts")
                return None

def get_total_pages(html_content):
    """Extract the total number of pages from the pagination information"""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Look for pagination elements
    pagination = soup.select("li.c-pagination__item")
    if pagination:
        # The last pagination item should contain the max page number
        try:
            last_page_link = pagination[-2].find("a")  # -2 to skip the "Next" button
            if last_page_link and last_page_link.text.strip().isdigit():
                return int(last_page_link.text.strip())
        except (IndexError, ValueError):
            pass
    
    # Fallback: try to find the total count and divide by items per page
    try:
        results_text = soup.select_one(".c-search-results__count")
        if results_text:
            total_items_match = re.search(r'(\d+,?\d*)\s+results', results_text.text)
            if total_items_match:
                total_items = int(total_items_match.group(1).replace(',', ''))
                items_per_page = 20  # Assuming 20 items per page
                return (total_items + items_per_page - 1) // items_per_page
    except Exception:
        pass
    
    # Default to the provided number if we can't determine it automatically
    return 8076

def parse_articles(html_content):
    """Parse articles from HTML content"""
    soup = BeautifulSoup(html_content, "html.parser")
    articles_data = []

    # Find all article cards
    article_cards = soup.find_all("div", class_="c-card__body u-display-flex u-flex-direction-column")

    for card in article_cards:
        article_info = {}

        # Extract title
        title_tag = card.find("h3", class_="c-card__title")
        if title_tag and title_tag.find("a"):
            article_info["title"] = title_tag.find("a").text.strip()
        else:
            article_info["title"] = None

        # Extract URL
        url_tag = card.find("h3", class_="c-card__title")
        if url_tag and url_tag.find("a"):
            article_info["url"] = "https://www.nature.com" + url_tag.find("a")["href"]
        else:
            article_info["url"] = None

        # Extract abstract
        abstract_tag = card.find("div", {"data-test": "article-description"})
        if abstract_tag and abstract_tag.find("p"):
            article_info["abstract"] = abstract_tag.find("p").text.strip()
        else:
            article_info["abstract"] = None
            
        # Extract Open Access
        open_access_span = card.find_next_sibling("div", class_="c-card__section c-meta")
        if open_access_span:
            open_access_tag = open_access_span.find("span", attrs={"data-test": "open-access"})
            if open_access_tag and "Open Access" in open_access_tag.text:
                article_info["open_access"] = "Y"
            else:
                article_info["open_access"] = "N"
        else:
            # Fallback if the c-meta section isn't a direct sibling or found this way
            # Check within the card itself as a last resort for any "Open Access" text
            # This part might need adjustment if the structure is different in some cards
            open_access_text_node = card.find(string=lambda text: text and "Open Access" in text)
            if open_access_text_node:
                article_info["open_access"] = "Y"
            else:
                article_info["open_access"] = "N"

        # Extract Date
        date_meta_section = card.find_next_sibling("div", class_="c-card__section c-meta")
        if date_meta_section:
            date_tag = date_meta_section.find("time", itemprop="datePublished")
            if date_tag and date_tag.has_attr("datetime"):
                article_info["date"] = date_tag["datetime"]
            elif date_tag: # Fallback to text content if datetime attribute is missing
                article_info["date"] = date_tag.text.strip()
            else:
                article_info["date"] = None
        else:
            article_info["date"] = None

        # Extract Authors
        authors_list = []
        # Authors are in a <ul> with data-test="author-list" which is a sibling of c-card__body
        # The example shows it inside a div that's a sibling, so let's find the parent article first
        # Each article is usually wrapped in an <article> tag or a main div container.
        # Assuming 'card' is the 'c-card__body', its parent might be 'c-card'.
        # Let's try to find the author list relative to the card's parent or siblings more robustly.
        
        # The provided snippet shows authors_ul is a sibling of the card's parent's sibling.
        # This means we need to navigate up and then find the specific ul.
        # Let's look for the <ul data-test="author-list" ...> within the broader article container.
        # The card itself is <div class="c-card__body ...">.
        # The author list <ul data-test="author-list"...> is a sibling to this div or a child of a sibling.
        
        # Corrected approach: The author list is a direct child of the 'card' div based on the new snippet.
        authors_ul = card.find("ul", attrs={"data-test": "author-list"})
        if authors_ul:
            author_tags = authors_ul.find_all("li", itemprop="creator")
            for li_tag in author_tags:
                name_span = li_tag.find("span", itemprop="name")
                if name_span:
                    authors_list.append(name_span.text.strip())
            if authors_list:
                article_info["author"] = ", ".join(authors_list)
            else:
                article_info["author"] = None
        else:
            article_info["author"] = None

        articles_data.append(article_info)
    
    return articles_data

class NatureScraper:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path("nature_data")
        self.base_url = "https://www.nature.com/nature/research-articles"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def run(self):
    
    all_articles = []
    
    from pathlib import Path
    
    # Create directory for saving results
    os.makedirs(self.output_dir, exist_ok=True)
    
    # Start with page 1
    url = f"{self.base_url}?searchType=journalSearch&sort=PubDate&page=1"
    first_page = fetch_page(url, self.headers)
    
    if not first_page:
        print("Failed to fetch the first page. Exiting.")
        return
    
    # Parse first page and get total pages
    total_pages = get_total_pages(first_page)
    print(f"Total pages detected: {total_pages}")
    
    # Define CSV headers based on the article structure
    csv_headers = ['title', 'url', 'abstract', 'open_access', 'date', 'author']
    
    # Process each page
    for page_num in range(1, total_pages + 1):
        print(f"Processing page {page_num} of {total_pages}...")
        
        # Only fetch the first page again if we're not on page 1
        if page_num == 1:
            html_content = first_page
        else:
            url = f"{self.base_url}?searchType=journalSearch&sort=PubDate&page={page_num}"
            html_content = fetch_page(url, self.headers)
            if not html_content:
                print(f"Skipping page {page_num} due to fetch error")
                continue
        
        # Parse articles from the page
        page_articles = parse_articles(html_content)
        all_articles.extend(page_articles)
        
        # Save intermediate results every 10 pages
        if page_num % 10 == 0 or page_num == total_pages:
            # Save page articles to CSV
            with open(self.output_dir / f"articles_page_{page_num}.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(page_articles)
            
            # Also update the complete dataset
            with open(self.output_dir / "all_articles.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(all_articles)
        
        # Be nice to the server - don't hammer it
        if page_num < total_pages:
            time.sleep(2)  # 2-second delay between requests
    
    print(f"Completed processing {len(all_articles)} articles across {total_pages} pages")
    print(f"Data saved to '{self.output_dir}/all_articles.csv'")

def main():
    scraper = NatureScraper()
    scraper.run()

if __name__ == "__main__":
    main() 
