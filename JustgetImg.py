import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
driver = None

def execChrome():
    global driver
    driver = webdriver.Chrome()
    driver.maximize_window()
    driver.get("https://map.kakao.com/?map_type=TYPE_MAP&map_attribute=ROADVIEW&panoid=1156738386&pan=233.1&tilt=15.9&zoom=0&urlLevel=3&urlX=462438&urlY=1102745")
    time.sleep(2)

def click2():
    global driver
    btn = driver.find_element(By.XPATH, '//*[@id="view"]/div[1]/div[1]/button')
    driver.execute_script("arguments[0].click();", btn) 
    time.sleep(2)
    print("1")
    btn = driver.find_element(By.XPATH, '//*[@id="view"]/div[1]/div[1]/ul/li[1]/button')
    driver.execute_script("arguments[0].click();", btn) 
    time.sleep(2)
    print("2")

execChrome()
click2()
time.sleep(10)