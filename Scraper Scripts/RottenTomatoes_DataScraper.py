from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import time
import pandas as pd

# --------------------------
# CONFIG
# --------------------------
driver = webdriver.Chrome()
driver.set_page_load_timeout(60)

# REMOVED NA TOP-CRITICS SINCE DUPED SIYA WITH ALL AUDIENCE
filters = ['all-audience', 'verified-audience', 'all-critics']
desired_reviews = 80  # per filter
wait = WebDriverWait(driver, 20)

all_reviews = []

# =============================
# SCRAPE AUDIENCE REVIEWS
# =============================
def scrape_audience():
    data = []
    
    try:
        # Find all review cards with multiple possible selectors
        review_selectors = [
            "review-card",
            "div.reviews-cards > review-card",
            "*[data-qa='review-card']",
            "review-card[data-qa='audience-review']"
        ]
        
        reviews = []
        for selector in review_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    reviews = elements
                    print(f"  Found {len(reviews)} reviews with selector: {selector}")
                    break
            except:
                continue
        
        if not reviews:
            # Fallback: Try to find any review-like elements
            all_elements = driver.find_elements(By.CSS_SELECTOR, "*")
            for elem in all_elements:
                if elem.tag_name == "review-card" or (hasattr(elem, 'get_attribute') and 
                                                     elem.get_attribute('data-qa') and 
                                                     'review' in elem.get_attribute('data-qa')):
                    reviews.append(elem)
            print(f"  Found {len(reviews)} reviews via fallback")
        
        for r in reviews:
            try:
                # 1. Get review text
                text = "No Review Text"
                try:
                    # Try multiple approaches to get full text
                    text_selectors = [
                        'drawer-more[slot="review"] span[slot="content"]',
                        'span[slot="content"]',
                        '.review-text',
                        'p.review-text',
                        'div[slot="review"]',
                        '*[data-qa="review-text"]'
                    ]
                    
                    for selector in text_selectors:
                        try:
                            text_elements = r.find_elements(By.CSS_SELECTOR, selector)
                            if text_elements:
                                text_element = text_elements[0]
                                text = text_element.text.strip()
                                
                                # If text is short, try clicking "See More"
                                if len(text) < 100:
                                    # Look for "See More" buttons
                                    see_more_selectors = [
                                        'rt-link[slot="cta-open"]',
                                        'button:contains("See More")',
                                        'a:contains("See More")',
                                        '*[data-qa="see-more"]'
                                    ]
                                    
                                    for see_more_selector in see_more_selectors:
                                        try:
                                            see_more_btn = r.find_element(By.CSS_SELECTOR, see_more_selector)
                                            if see_more_btn and see_more_btn.is_displayed():
                                                driver.execute_script("arguments[0].click();", see_more_btn)
                                                time.sleep(0.3)
                                                # Re-get text after expansion
                                                text_elements = r.find_elements(By.CSS_SELECTOR, selector)
                                                if text_elements:
                                                    text = text_elements[0].text.strip()
                                                break
                                        except:
                                            continue
                                
                                if text and text != "No Review Text":
                                    break
                        except:
                            continue
                except:
                    pass

                # 2. Get name
                name = "Anonymous"
                try:
                    name_selectors = [
                        'rt-link[slot="name"]',
                        '.display-name',
                        '.reviewer-name',
                        '*[data-qa="reviewer-name"]',
                        'span[slot="name"]'
                    ]
                    
                    for selector in name_selectors:
                        try:
                            name_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if name_elem:
                                name = name_elem.text.strip()
                                if not name:
                                    # Try to get from href
                                    href = name_elem.get_attribute('href')
                                    if href and '/profiles/' in href:
                                        name = href.split('/profiles/')[-1].split('/')[0]
                                if name:
                                    break
                        except:
                            continue
                except:
                    pass

                # 3. Get date
                date = "No Date"
                try:
                    date_selectors = [
                        'span[slot="timestamp"]',
                        '.review-date',
                        '.timestamp',
                        '*[data-qa="review-date"]',
                        'time'
                    ]
                    
                    for selector in date_selectors:
                        try:
                            date_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if date_elem:
                                date = date_elem.text.strip()
                                if date:
                                    break
                        except:
                            continue
                except:
                    pass

                # 4. Get rating
                rating = "No Rating"
                try:
                    rating_selectors = [
                        'rating-stars-group[slot="rating"]',
                        '.rating',
                        '.score',
                        '*[data-qa="review-score"]',
                        '*[score]'
                    ]
                    
                    for selector in rating_selectors:
                        try:
                            rating_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if rating_elem:
                                # Try different ways to get the score
                                score = rating_elem.get_attribute('score')
                                if score:
                                    try:
                                        score_num = float(score)
                                        rating = f"{score_num}/5"
                                    except:
                                        rating = f"{score}/5"
                                else:
                                    rating_text = rating_elem.text.strip()
                                    if rating_text:
                                        rating = rating_text
                                if rating != "No Rating":
                                    break
                        except:
                            continue
                except:
                    pass

                data.append({
                    "Type": "Audience",
                    "Filter": "Audience",
                    "Review Content": text,
                    "Name": name,
                    "Date": date,
                    "Rating": rating
                })
                
            except StaleElementReferenceException:
                print(" Stale element, skipping...")
                continue
            except Exception as e:
                print(f" Error processing a review: {e}")
                continue
    
    except Exception as e:
        print(f"  ❌ Error in scrape_audience: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  Successfully extracted {len(data)} audience reviews")
    return data

# =============================
# SCRAPE CRITIC REVIEWS
# =============================
def scrape_critics():
    data = []
    
    try:
        # Find all critic reviews
        review_selectors = [
            "review-speech-balloon",
            "review-card[top-critic]",
            "*[data-qa='critic-review']",
            "div.reviews-cards > review-speech-balloon"
        ]
        
        reviews = []
        for selector in review_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    reviews = elements
                    print(f"  Found {len(reviews)} critic reviews with selector: {selector}")
                    break
            except:
                continue
        
        if not reviews:
            print("  No critic reviews found with selectors")
            return data
        
        for r in reviews:
            try:
                tag_name = r.tag_name
                
                # Get review text
                text = "No Review Text"
                try:
                    text_selectors = [
                        'p.review-text',
                        '.review-text-container',
                        'drawer-more span[slot="content"]',
                        '.review-body',
                        '*[data-qa="review-text"]'
                    ]
                    
                    for selector in text_selectors:
                        try:
                            text_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if text_elem:
                                text = text_elem.text.strip()
                                if text:
                                    # Try to expand if needed
                                    if len(text) < 50:
                                        try:
                                            see_more_btn = r.find_element(By.CSS_SELECTOR, 'button:contains("Read Full Review")')
                                            if see_more_btn and see_more_btn.is_displayed():
                                                driver.execute_script("arguments[0].click();", see_more_btn)
                                                time.sleep(0.3)
                                                text_elem = r.find_element(By.CSS_SELECTOR, selector)
                                                text = text_elem.text.strip()
                                        except:
                                            pass
                                    break
                        except:
                            continue
                except:
                    pass

                # Get critic name
                name = "No Name"
                try:
                    name_selectors = [
                        '.critic-name',
                        '.display-name',
                        '*[data-qa="critic-name"]',
                        'rt-link[slot="name"]'
                    ]
                    
                    for selector in name_selectors:
                        try:
                            name_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if name_elem:
                                name = name_elem.text.strip()
                                if name:
                                    break
                        except:
                            continue
                except:
                    pass

                # Get date
                date = "No Date"
                try:
                    date_selectors = [
                        '.review-date',
                        '.date',
                        '*[data-qa="review-date"]',
                        'span[slot="timestamp"]'
                    ]
                    
                    for selector in date_selectors:
                        try:
                            date_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if date_elem:
                                date = date_elem.text.strip()
                                if date:
                                    break
                        except:
                            continue
                except:
                    pass

                # Get rating/score
                rating = "No Score"
                try:
                    rating_selectors = [
                        '.icon',
                        '.score',
                        '*[data-qa="review-score"]',
                        '*[data-rating]',
                        'rating-stars-group[slot="rating"]'
                    ]
                    
                    for selector in rating_selectors:
                        try:
                            rating_elem = r.find_element(By.CSS_SELECTOR, selector)
                            if rating_elem:
                                score = rating_elem.text.strip()
                                if not score:
                                    score = rating_elem.get_attribute('data-rating')
                                    if not score:
                                        score = rating_elem.get_attribute('score')
                                if score:
                                    rating = score
                                    break
                        except:
                            continue
                except:
                    pass

                data.append({
                    "Type": "Critic",
                    "Filter": "Critic",
                    "Review Content": text,
                    "Name": name,
                    "Date": date,
                    "Rating": rating
                })
                
            except StaleElementReferenceException:
                print("  Stale element, skipping...")
                continue
            except Exception as e:
                print(f"  Error processing critic review: {e}")
                continue
    
    except Exception as e:
        print(f" Error in scrape_critics: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  Successfully extracted {len(data)} critic reviews")
    return data

# =============================
# IMPROVED: LOAD MORE FUNCTION
# =============================
def click_load_more():
    """Try to find and click the load more button"""
    button_selectors = [
        "button[data-qa='load-more-btn']",
        "rt-button[data-qa='load-more-btn']",
        "button.load-more",
        "button.show-more-btn",
        "button:contains('Load More')",
        "button:contains('Show More')",
        "rt-button:contains('Load More')",
        "a:contains('Load More')"
    ]
    
    for selector in button_selectors:
        try:
            if "contains" in selector:
                text = "Load More" if "Load More" in selector else "Show More"
                buttons = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            else:
                buttons = driver.find_elements(By.CSS_SELECTOR, selector)
            
            for button in buttons:
                try:
                    if button.is_displayed() and button.is_enabled():
                        # Scroll to button
                        driver.execute_script("""
                            arguments[0].scrollIntoView({
                                behavior: 'smooth',
                                block: 'center',
                                inline: 'center'
                            });
                        """, button)
                        time.sleep(1)
                        
                        # Try JavaScript click
                        driver.execute_script("arguments[0].click();", button)
                        print(f"  Clicked 'Load More' button using selector: {selector}")
                        time.sleep(2)  # Wait for content to load
                        return True
                except:
                    continue
        except:
            continue
    
    # Also try to scroll to trigger lazy loading
    try:
        # Scroll to bottom to trigger lazy loading if any
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        print("  Scrolled to bottom to trigger loading")
        time.sleep(2)
        return True
    except:
        pass
    
    return False

# =============================
# WAIT FOR REVIEWS TO LOAD
# =============================
def wait_for_reviews(timeout=10):
    """Wait for reviews to be present on the page"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check for any review elements
            review_selectors = [
                "review-card",
                "review-speech-balloon",
                "*[data-qa*='review']",
                ".reviews-cards > *"
            ]
            
            for selector in review_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements and len(elements) > 0:
                        print(f"  ✅ Found {len(elements)} review elements")
                        return True
                except:
                    continue
            
            time.sleep(0.5)
        except:
            time.sleep(0.5)
    
    print("  ⚠️ Timeout waiting for reviews")
    return False

# =============================
# MAIN LOOP
# =============================
for f in filters:
    url = f"https://www.rottentomatoes.com/m/you_animal/reviews/{f}"
    print(f"\n{'='*60}")
    print(f"Processing: {url}")
    print(f"{'='*60}")
    
    try:
        # Load page with retry
        for attempt in range(3):
            try:
                driver.get(url)
                # Wait for page to be fully loaded
                WebDriverWait(driver, 20).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"  Attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        
        time.sleep(3)
        
        # Check for empty state (e.g., no reviews found)
        try:
            empty_state = driver.find_element(By.CSS_SELECTOR, "rt-text[data-qa='no-reviews-text']")
            if "No " in empty_state.text and "reviews found" in empty_state.text:
                print(f"  {empty_state.text}, skipping {f}")
                continue
        except:
            pass
        
        # Wait for reviews to load initially
        if not wait_for_reviews():
            print(f"  No reviews found on initial load for {f}")
            continue
        
        collected = 0
        load_more_attempts = 0
        max_attempts = 15  # Increased for more scrolling
        previous_count = 0
        no_new_count = 0
        
        while collected < desired_reviews and load_more_attempts < max_attempts:
            # Scrape reviews
            if "audience" in f:
                new_reviews = scrape_audience()
            else:
                new_reviews = scrape_critics()
            
            if not new_reviews:
                print("  No reviews scraped")
                break
            
            # Count only new reviews
            new_reviews_count = len(new_reviews)
            if new_reviews_count == previous_count:
                no_new_count += 1
                if no_new_count >= 3:
                    print(" No new reviews loaded for 3 attempts, stopping")
                    break
            else:
                no_new_count = 0
                previous_count = new_reviews_count
            
            # Add to collection with unique IDs
            for review in new_reviews:
                # Create a unique ID for each review
                review_id = f"{f}_{review['Name']}_{review['Date']}_{hash(review['Review Content'][:100])}"
                review["Review_ID"] = review_id
                
                # Check if we already have this review
                if not any(r.get("Review_ID") == review_id for r in all_reviews):
                    review["Source Filter"] = f
                    all_reviews.append(review)
                    collected += 1
            
            print(f" Filter '{f}': Collected {collected} unique reviews")
            
            if collected >= desired_reviews:
                break
            
            # Try to load more
            print(f"  Attempting to load more reviews ({load_more_attempts + 1}/{max_attempts})...")
            if click_load_more():
                load_more_attempts += 1
                time.sleep(2)  # Wait for new content
            else:
                print(" No more content to load")
                break
        
        print(f" Completed filter '{f}', collected {collected} unique reviews")
        
    except Exception as e:
        print(f"  Error processing {f}: {e}")
        import traceback
        traceback.print_exc()

# Close browser
driver.quit()

# Save results
if all_reviews:
    df = pd.DataFrame(all_reviews)
    
    # Remove duplicates based on Review_ID
    df = df.drop_duplicates(subset=['Review_ID'], keep='first')
    
    # Remove the temporary ID column
    if 'Review_ID' in df.columns:
        df = df.drop(columns=['Review_ID'])
    
    # Save to files
    excel_path = r"C:\Users\Josh\Downloads\PROJ-C-BREAK\Data\rottentomatoes\rottentomatoes_reviews.xlsx"
    csv_path = r"C:\Users\Josh\Downloads\PROJ-C-BREAK\Data\rottentomatoes\rottentomatoes_reviews.csv"
    
    df.to_excel(excel_path, index=False)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully saved {len(df)} unique reviews")
    print(f"Excel file: {excel_path}")
    print(f"CSV file: {csv_path}")
    
    # Show statistics
    print("\nReview statistics:")
    print(df['Type'].value_counts())
    print(f"\nTotal unique reviews: {len(df)}")
    print(f"{'='*60}")
else:
    print("\nNo reviews collected")