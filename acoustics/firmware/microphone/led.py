from machine import Pin, PWM
import machine
import time
# Define pins for R, G, B
red = PWM(Pin(17), freq=500)
green = PWM(Pin(4), freq=500)
blue = PWM(Pin(16), freq=500)

def set_color(r, g, b):
    # Values: 0 (full brightness) to 1023 (off)
    red.duty(1023 - r)
    green.duty(1023 - g)
    blue.duty(1023 - b)

def on_error():
    print("On Error")
    for i in range(10):
        red.duty(0)
        time.sleep(1)
        red.duty(1023)
        time.sleep(1)
    machine.reset()
        

#set_color(1023, 0, 0) # white
#set_color(0, 1023, 0)   # Blue
#set_color(0, 0, 1023)   # Green
#set_color(0, 100, 100)      # light blue
#set_color(100,100,100) # white



