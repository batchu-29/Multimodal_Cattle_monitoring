# main.py
import time
import json
import ubinascii
import machine
from machine import Pin, I2S
from umqtt.simple import MQTTClient
from led import set_color,on_error

# Fixed metadata
BOVINE_ID = "2"
MQTT_BROKER = "192.168.0.3"
MQTT_PORT = 1883
MQTT_TOPIC = b"inference/microphone"


# Audio config
CHUNK_SIZE = 1024
SAMPLE_RATE = 22500 
BITS_PER_SAMPLE = 16 
CHANNELS = 1
DURATION_SEC = 5
MOCK_MQTT = "broker.hivemq.com"
TOTAL_BYTES = SAMPLE_RATE * (BITS_PER_SAMPLE // 8) * CHANNELS * DURATION_SEC
NUM_CHUNKS = TOTAL_BYTES // CHUNK_SIZE + (1 if TOTAL_BYTES % CHUNK_SIZE else 0)

#print(f"Audio will be split into {NUM_CHUNKS} chunks")

def mqtt_connect(client_id="esp3-client", server = MQTT_BROKER, port=1883, max_retries=5,use_mock=False):
    attempt = 0
    print("attempting to connect to mqtt")
    if(use_mock):
        server = MOCK_MQTT
    while attempt < max_retries:
        try:
            client = MQTTClient(client_id, server, port)
            client.connect()
            print("✅ Connected to MQTT broker")
            return client
        except Exception as e:
            print(f"❌ MQTT connect failed: {e}")
            delay = 1 << attempt
            print(f"⏳ Retrying in {delay}s...")
            time.sleep(5)
            attempt += 1
    raise RuntimeError("Failed to connect to MQTT broker")



import time

def iso8601_now(time):
    return "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}".format(*time[:6])

def ist_time():
    return time.localtime(time.time() + 19800)  # UTC + 5h30m = 19800 sec


def on_button_press():
    try:
        buf = bytearray(CHUNK_SIZE)
        print("Recording 5s of audio and sending...",iso8601_now(ist_time()))
        audio_data = []

            # Send start chunk
        start_msg = json.dumps({
                "bovine_id": BOVINE_ID,
                "type": "start",
                "timestamp": iso8601_now(ist_time()),
                "chunks": NUM_CHUNKS
        })
        client.publish(MQTT_TOPIC, start_msg)

            # Read and send chunks
        chunk_index = 0
        bytes_left = TOTAL_BYTES

        while bytes_left > 0:
            read_len = min(CHUNK_SIZE, bytes_left)
            buf = bytearray(read_len)
            i2s.readinto(buf)
            encoded = ubinascii.b2a_base64(buf).decode().strip()

            data_msg = json.dumps({
                    "bovine_id": BOVINE_ID,
                    "type": "data",
                    "index": chunk_index,
                    "data": encoded
            })
            client.publish(MQTT_TOPIC, data_msg)

            chunk_index += 1
            bytes_left -= read_len

            # Send end chunk
        end_msg = json.dumps({
                "bovine_id": BOVINE_ID,
                "type": "end",
                "timestamp": iso8601_now(ist_time())
        })
        client.publish(MQTT_TOPIC, end_msg)
        set_color(0, 100, 100)      # light blue
        print("Finished sending one 5s audio session....Sleeping for 5s")
        
        time.sleep(5)

    finally:
        pass

from machine import Pin, Timer
import time

# Configuration
BOOT_PIN = 0            # GPIO0 is usually the BOOT button
DEBOUNCE_MS = 200       # Debounce time in milliseconds

# Globals
last_pressed = 0
debounce_timer = Timer(0)  # Use a real timer ID (0 or 1 on ESP8266, 0-3 on ESP32)

def user_function():
    """Function triggered by button press."""
    print("BOOT button pressed! Executing action...")
    set_color(100, 100, 100)      # light yellow
    on_button_press()
    print("Button Action Complete")
    set_color(0, 0, 1023)   # Green
    

def debounce_callback(timer):
    """Validate the press after debounce delay."""
    now = time.ticks_ms()
    if time.ticks_diff(now, last_pressed) >= DEBOUNCE_MS:
        user_function()

def button_irq(pin):
    """Handle button press with debounce timer."""
    global last_pressed
    last_pressed = time.ticks_ms()
    try:
        debounce_timer.deinit()  # Clean up before reinitializing
    except:
        pass  # May already be stopped
    debounce_timer.init(mode=Timer.ONE_SHOT, period=DEBOUNCE_MS, callback=debounce_callback)

def setup():
    """Initialize the button and IRQ."""
    button = Pin(BOOT_PIN, Pin.IN, Pin.PULL_UP)
    button.irq(trigger=Pin.IRQ_FALLING, handler=button_irq)
    print("Button handler set up on GPIO{}".format(BOOT_PIN))
    

        
def listen_for_button_press():
    global client
    global i2s
    # I2S Microphone config (INMP441)
    i2s = I2S(
        0,
        sck=Pin(23), 
        ws=Pin(22),
        sd=Pin(21),
        mode=I2S.RX,
        bits=16,
        format=I2S.MONO,
        rate=22500,
        ibuf=8192
    )
    client = mqtt_connect()
    setup()
    set_color(0, 0, 1023)   # Green
    
