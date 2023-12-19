import serial
import struct
import time

HOVER_SERIAL_BAUD = 115200
SERIAL_BAUD = 115200
START_FRAME = 0xABCD
TIME_SEND = 100
SPEED_MAX_TEST = 300
SPEED_STEP = 20

serial_port = "/dev/ttyS0"  # Change to the correct port

ser = serial.Serial(serial_port, SERIAL_BAUD, timeout=1)

def send_data(steer, speed):
    start_frame = struct.pack('<H', START_FRAME)
    steer_data = struct.pack('<h', steer)
    speed_data = struct.pack('<h', speed)

    start_frame_value, = struct.unpack('<H', start_frame)
    steer_value, = struct.unpack('<h', steer_data)
    speed_value, = struct.unpack('<h', speed_data)
	#print(struct.pack('<I', start_frame_value ^ steer_value ^ speed_value))
    checksum = struct.pack('<H', (start_frame_value ^ steer_value ^ speed_value) & 0xFFFF)

    message = start_frame + steer_data + speed_data + checksum
    ser.write(message)

try:
    while True:
        send_data(50, -50)
        time.sleep(0.1)

except KeyboardInterrupt:
    ser.close()


