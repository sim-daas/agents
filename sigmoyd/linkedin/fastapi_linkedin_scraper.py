from fastapi import FastAPI, Query
from typing import List
import uvicorn
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

app = FastAPI()

def scrape_linkedin_posts(search_query: str, quantity: int = 10) -> List[str]:
    job_title = search_query
    to_ret = []
    with webdriver.Firefox() as driver:
        driver.get(f"https://www.linkedin.com")
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

@app.get("/search")
def search_linkedin(query: str = Query(..., description="Search query for LinkedIn posts"), quantity: int = 10):
    results = scrape_linkedin_posts(query, quantity)
    return {"results": results}

# Uncomment below to run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
