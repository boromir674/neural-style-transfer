#!/usr/bin/env python
import os

from selenium import webdriver

# from selenium.webdriver.common.keys import Keys

my_dir = os.path.dirname(os.path.realpath(__file__))
driver = webdriver.Chrome(os.path.join(my_dir, "../chromedriver"))

driver.get("https://drive.protonmail.com/urls/7RXGN23ZRR#hsw4STil0Hgc")

print(driver.title)

# search_bar = driver.find_element_by_name("q")
# search_bar.clear()
# search_bar.send_keys("getting started with python")
# search_bar.send_keys(Keys.RETURN)

print(driver.current_url)

driver.close()
