from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import pickle

# Parameters
quantity = 10
job_title = "python"

# Set up Firefox driver
with webdriver.Firefox() as driver:
    driver.get(f"https://www.linkedin.com")
    # Load cookies from file
    with open('/root/linkedin_cookies.pkl', 'rb') as f:
        cookies = pickle.load(f)
    for cookie in cookies:
        # Remove 'sameSite' if present, as it may cause issues
        cookie.pop('sameSite', None)
        driver.add_cookie(cookie)
    driver.refresh()
    time.sleep(2)

    # Go to LinkedIn search for posts with keywords
    search_url = f"https://www.linkedin.com/search/results/content/?keywords=%22hiring%22%20%26%20%22AI%22%20%26%20%22{job_title}%22&origin=GLOBAL_SEARCH_HEADER"
    driver.get(search_url)
    time.sleep(3)

    # Scroll down multiple times to load more posts
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    # Expand all "See more" buttons to get full posts
    see_more_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'more')]")
    for button in see_more_buttons:
        try:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(1)
        except:
            pass

    # Extract full post contents
    posts = driver.find_elements(By.CSS_SELECTOR, ".feed-shared-update-v2__description, .update-components-text")
    to_ret = []
    for index, post in enumerate(posts[:quantity]):
        print(f"\n🔹 Post {index + 1}:\n{post.text}\n{'-'*50}")
        to_ret.append(post.text)

    print(to_ret)