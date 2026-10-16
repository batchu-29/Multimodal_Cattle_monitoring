# boot.py
import network
import time
import ntptime
from main import listen_for_button_press
from machine import Pin, PWM
from time import sleep
from led import set_color,on_error

SSID = "YOUR_WIFI_SSID"
PASSWORD = "YOUR_WIFI_PASSWORD"
def wifi_connect(ssid, password, max_retries=3):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    attempt = 0

    while not wlan.isconnected() and attempt < max_retries:
        print(f"ðŸ“¡ Attempting Wi-Fi connection ({attempt + 1})...")
        try:
            wlan.connect(ssid, password)
        except Exception as e:
            print(e)
            set_color(1023, 1023, 0)   # Red
        timeout = 1 << attempt  # exponential backoff (2^attempt seconds)
        for _ in range(timeout):
            if wlan.isconnected():
                break
            time.sleep(1)
        attempt += 1
    
    try:
        if wlan.isconnected():
            ntptime.settime()
    except:
        print("Failed to set time")
        
    print(time.localtime())

    if wlan.isconnected():
        print("Connected to Wi-Fi:", wlan.ifconfig()[0])
    else:
        on_error()
        print("Failed to connect to Wi-Fi")
    return wlan
        
    

set_color(100,100,100) # White/Yelllow
wlan = wifi_connect(SSID, PASSWORD)
if wlan.isconnected():
    listen_for_button_press()


