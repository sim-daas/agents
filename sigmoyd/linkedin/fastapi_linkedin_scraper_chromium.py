from fastapi import FastAPI, Query
from typing import List
import uvicorn
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor
import time

app = FastAPI()

def scrape_linkedin_posts(search_query: str, quantity: int = 10) -> List[str]:
    job_title = search_query
    to_ret = []
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--max_old_space_size=4096")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    
    with webdriver.Chrome(options=chrome_options) as driver:
        driver.get("https://www.linkedin.com")
        with open('/home/admin/agents/sigmoyd/linkedin/linkedin_cookies.pkl', 'rb') as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            cookie.pop('sameSite', None)
            driver.add_cookie(cookie)
        driver.refresh()
        time.sleep(2)
        search_url = f"https://www.linkedin.com/search/results/content/?keywords=%22hiring%22%20%26%20%22AI%22%20%26%20%22{job_title}%22&origin=GLOBAL_SEARCH_HEADER"
        driver.get(search_url)
        time.sleep(3)
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
        see_more_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'more')]")
        for button in see_more_buttons:
            try:
                driver.execute_script("arguments[0].click();", button)
                time.sleep(1)
            except:
                pass
        posts = driver.find_elements(By.CSS_SELECTOR, ".feed-shared-update-v2__description, .update-components-text")
        for post in posts[:quantity]:
            to_ret.append(post.text)
    return to_ret

def extract_company_info(link: str) -> str:
    """Extract company information from LinkedIn profile page using headless Chrome"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--max_old_space_size=4096")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    
    with webdriver.Chrome(options=chrome_options) as driver:
        try:
            driver.get("https://www.linkedin.com")
            with open('/home/admin/agents/sigmoyd/linkedin/linkedin_cookies.pkl', 'rb') as f:
                cookies = pickle.load(f)
            for cookie in cookies:
                cookie.pop('sameSite', None)
                driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(2)
            
            driver.get(link + "/about/")
            wait = WebDriverWait(driver, 10)
            
            try:
                about_section = wait.until(EC.presence_of_element_located((
                    By.CSS_SELECTOR, 
                    "section.artdeco-card.org-page-details-module__card-spacing.artdeco-card.org-about-module__margin-bottom"
                )))
                about_text = about_section.text if about_section else "About section not found"
                return about_text
            except Exception as e:
                return f"Error extracting company about section: {str(e)}"
        except Exception as e:
            return f"Error processing {link}: {str(e)}"

def search_linkedin_companies(name_or_category: str, countries: List[str] = None, industries: List[str] = None) -> List[str]:
    """Search LinkedIn companies with filters using headless Chrome"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--memory-pressure-off")
    chrome_options.add_argument("--max_old_space_size=4096")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    
    with webdriver.Chrome(options=chrome_options) as driver:
        try:
            driver.get("https://www.linkedin.com")
            with open('/home/admin/agents/sigmoyd/linkedin/linkedin_cookies.pkl', 'rb') as f:
                cookies = pickle.load(f)
            for cookie in cookies:
                cookie.pop('sameSite', None)
                driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(2)
            
            driver.get(f"https://www.linkedin.com/search/results/companies/?keywords={name_or_category}")
            wait = WebDriverWait(driver, 10)
            time.sleep(3)

            if countries or industries:
                try:
                    # Try multiple methods to find the All filters button using exact selectors
                    all_filters_button = None
                    
                    # Method 1: Try by exact class selector
                    try:
                        all_filters_button = driver.find_element(By.CSS_SELECTOR, "button.search-reusables__all-filters-pill-button")
                    except:
                        pass
                    
                    # Method 2: Try by aria-label
                    if not all_filters_button:
                        try:
                            all_filters_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label*='Show all filters']")
                        except:
                            pass
                    
                    # Method 3: Try by multiple class combination
                    if not all_filters_button:
                        try:
                            all_filters_button = driver.find_element(By.CSS_SELECTOR, "button.artdeco-pill.search-reusables__filter-pill-button")
                        except:
                            pass
                    
                    # Method 4: Try by ID pattern (ember IDs change but we can try)
                    if not all_filters_button:
                        try:
                            all_filters_button = driver.find_element(By.CSS_SELECTOR, "button[id^='ember'][aria-label*='Show all filters']")
                        except:
                            pass
                    
                    # Method 5: Try by text content
                    if not all_filters_button:
                        try:
                            buttons = driver.find_elements(By.TAG_NAME, "button")
                            for button in buttons:
                                if "All filters" in button.text:
                                    all_filters_button = button
                                    break
                        except:
                            pass
                    
                    if all_filters_button:
                        driver.execute_script("arguments[0].click();", all_filters_button)
                        time.sleep(3)
                        print("Successfully clicked All filters button")
                    else:
                        print("Could not find All filters button, skipping filters")

                    if countries:
                        for country in countries:
                            try:
                                country_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Add a location')]")))
                                country_button.click()
                                time.sleep(1)
                                
                                country_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[contains(@placeholder, 'Add a location')]")))
                                country_input.clear()
                                country_input.send_keys(country)
                                time.sleep(2)
                                
                                country_input.send_keys(Keys.ARROW_DOWN)
                                country_input.send_keys(Keys.RETURN)
                            except Exception as e:
                                print(f"Error adding country {country}: {e}")

                    if industries:
                        for industry in industries:
                            try:
                                industry_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Add an industry')]")))
                                industry_button.click()
                                time.sleep(1)
                                
                                industry_input = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[contains(@placeholder, 'Add an industry')]")))
                                industry_input.clear()
                                industry_input.send_keys(industry)
                                time.sleep(2)
                                
                                industry_input.send_keys(Keys.ARROW_DOWN)
                                industry_input.send_keys(Keys.RETURN)
                            except Exception as e:
                                print(f"Error adding industry {industry}: {e}")

                    show_results_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Show results')]")))
                    show_results_button.click()
                    time.sleep(3)
                except Exception as e:
                    print(f"Error applying filters: {e}")
            
            company_containers = driver.find_elements(By.CSS_SELECTOR, "div.wXrwwRbEXDlojRuemfAPZdgmGApJnxZLAQ")
            
            links = []
            for container in company_containers:
                try:
                    company_element = container.find_element(By.CSS_SELECTOR, "a.SiHrjwqVjdhQxOAoJFSxYpQQWnHauKpeg")
                    company_url = company_element.get_attribute("href")
                    links.append(company_url)
                except Exception as e:
                    print(f"Failed to extract company from container: {e}")
            
            return links
        except Exception as e:
            return [f"Error searching companies: {str(e)}"]

@app.get("/search")
def search_linkedin(query: str = Query(..., description="Search query for LinkedIn posts"), quantity: int = 10):
    results = scrape_linkedin_posts(query, quantity)
    return {"results": results}

@app.get("/company")
def get_company_info(link: str = Query(..., description="LinkedIn company profile URL")):
    result = extract_company_info(link)
    return {"company_info": result}

@app.get("/companies")
def search_companies(
    name_or_category: str = Query(..., description="Company name or category to search for"),
    countries: List[str] = Query(None, description="List of countries to filter by"),
    industries: List[str] = Query(None, description="List of industries to filter by")
):
    results = search_linkedin_companies(name_or_category, countries or [], industries or [])
    return {"company_links": results}

# Uncomment below to run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)