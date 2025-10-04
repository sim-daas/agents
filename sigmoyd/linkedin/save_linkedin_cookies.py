from selenium import webdriver
import time
import pickle

# Open LinkedIn and wait for manual login
with webdriver.Firefox() as driver:
    driver.get('https://www.linkedin.com/')
    print('Please log in to LinkedIn manually in the opened browser window.')
    time.sleep(60)  # Give user time to log in
    cookies = driver.get_cookies()
    with open('/root/linkedin_cookies.pkl', 'wb') as f:
        pickle.dump(cookies, f)
    print('Cookies saved to /root/linkedin_cookies.pkl')



'''

    driver.get('https://www.nvidia.com/en-in/training/')
    # Wait for the search box to be present
    wait = WebDriverWait(driver, 15)
    # Accept cookies if the button is present
    try:
        accept_btn = wait.until(EC.element_to_be_clickable((By.ID, 'onetrust-accept-btn-handler')))
        accept_btn.click()
        time.sleep(random.uniform(0.5, 1.2))
    except Exception:
        pass  # If not present, continue
    # Click the search icon using By.ID
    search_icon = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "li#nv-search-box a.menu-level-1"))
    )
    search_icon.click() # Wait for the search text box to appear using By.ID
    search_box = wait.until(EC.presence_of_element_located((By.ID, 'search-terms')))
    time.sleep(random.uniform(1.2, 2.5))
    # Simulate human typing
    for char in 'nvidia':
        search_box.send_keys(char)
        time.sleep(random.uniform(0.1, 0.3))
    search_box.send_keys(Keys.RETURN)
    # Realistic delay before closing
    time.sleep(random.uniform(5, 10))
    '''