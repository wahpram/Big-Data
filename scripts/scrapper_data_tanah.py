from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from time import sleep
from seleniumbase import Driver
from load_data_to_mongo import load_to_mongo
import csv
import os
import datetime


def load_page(self, element):
    global myElem
    delay = 5
    try:
        myElem = WebDriverWait(self, delay).until(EC.presence_of_element_located((By.XPATH, element)))
    except TimeoutException:
        print('Loading too much time')

    return myElem


def get_text_with_class(self, element, default=""):
    global myElem
    delay = 2
    try:
        myElem = WebDriverWait(self, delay).until(EC.presence_of_element_located((By.CSS_SELECTOR, f".{element}")))
        return myElem.text
    except Exception:
        print('Loading too much time')
        return default


def main():
    data = dict()
    driver = Driver(uc=True)
    
    col_name = 'raw'
    
    try:
        for i in range(1, 1344+1):
            driver.get(f"https://www.rumah123.com/jual/bali/tanah/?sort=posted-desc&page={i}")
            
            element = driver.find_element(By.CSS_SELECTOR, '.ui-search-page__content.relative.ui-col-12')
            
            cards = element.find_elements(By.CSS_SELECTOR, '.ui-organism-intersection__element.intersection-card-container')
            
            for _id, card in enumerate(cards):
                try: 
                    _id = f'{i}_{_id}'
                    
                    card_feature = card.find_element(By.CSS_SELECTOR, '.card-featured__content-wrapper')
                    price = card_feature.find_element(By.CSS_SELECTOR, '.card-featured__middle-section__price')
                    price = price.text
                    
                    title = card_feature.find_element(By.TAG_NAME, 'a')
                    title = title.text
                    
                    span_elements = card_feature.find_elements(By.CSS_SELECTOR, 'span')
                    span_texts = [span.text for span in span_elements]
                    address = span_texts[1]
                    
                    land_area = card_feature.find_element(By.CSS_SELECTOR, '.attribute-info')
                    land_area = land_area.text
                    cleaned_land_area = land_area.replace("LT : ", "")
                    
                    card_bottom = card.find_element(By.CSS_SELECTOR, '.ui-organisms-card-r123-basic__bottom-section')
                    bottom_hidden = card_bottom.find_element(By.CSS_SELECTOR, '.ui-organisms-card-r123-basic__bottom-section__agent')
                    div = bottom_hidden.find_element(By.TAG_NAME, 'div')
                    date = div.find_element(By.TAG_NAME, 'p')
                    date = date.text
                    
                    print(f'_id = {_id}, Title: {title}, Date: {date}, Price: {price}, Address: {address}, Land area: {cleaned_land_area}')
                    
                    
                    data[_id] = {
                        'title' : title,
                        'date' : date,
                        'price' : price,
                        'address' : address,
                        'land_area' : cleaned_land_area,
                        'timestamp' : datetime.datetime.now().replace(microsecond=0)
                    }
                
                except:
                    pass

            print(f"-----------------Halaman ke-{i}-----------------")
        load_to_mongo(data, col_name)
        
    
    except Exception as e:
        print('Error :(')
        print(f'{e}')
        load_to_mongo(data, col_name)
    
    print(f'Scrape data telah berhasil dan selesai :D') 
    
    sleep(10)
    
            
if __name__ == "__main__":
    main()