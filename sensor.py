from gpiozero import DistanceSensor, LED
led = LED(1)

led.on()

s2 = DistanceSensor(echo=24, trigger=23)
s1 = DistanceSensor(echo=20, trigger=21)

while True:
	print(s1.distance, '   ', s2.distance)
