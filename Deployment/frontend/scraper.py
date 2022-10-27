from playwright.sync_api import sync_playwright
import pandas as pd

def scrape(link):
    URL = link

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50) # ,proxy=PROXIES_ID
        page = browser.new_page()
        page.set_default_timeout(300000)
        page.goto(URL)
        page.click('div > label:nth-child(5) > div > div > span')
        page.click('div > label:nth-child(4) > div > div > span')
        page.click('div > label:nth-child(3) > div > div > span')
        
        page.wait_for_timeout(2000) # settings for scraping model default = 2000, for webapp default = 3000

        review_list = []

        for i in range(5):
            review_raw = page.query_selector_all('div:nth-child(1) > p:nth-child(4) > span:nth-child(1)')
            for x in review_raw:
                replaced = x.inner_text().replace("\n", "")
                review_list.append(replaced)
            
            next_page = page.locator('role=button[name="Laman berikutnya"]')

            if next_page.is_visible():
                if next_page.is_disabled() == False:
                    next_page.click()
                    page.wait_for_timeout(2000) # settings for scraping model default = 2000, for webapp default = 3000
                else:
                    break
            else:
                break

        review_dict = {
            "text" : review_list
        }
        review_df = pd.DataFrame.from_dict(review_dict)
        return review_df
        # print(review_df)

        
        # ~ print(dada.inner_text()) # debugging
