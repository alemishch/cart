import RPi.GPIO as GPIO          
import time

en1 = 19
en2 = 12
en3 = 18
en4 = 13
in1_1 = 26
in1_2 = 6
in2_1 = 5
in2_2 = 0
in3_1 = 20
in3_2 = 16
in4_1 = 1
in4_2 = 7


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

class Motor:
	def __init__(self, in1, in2, en):
		self.in1 = in1
		self.in2 = in2
		self.en = en
		GPIO.setup(self.in1, GPIO.OUT)
		GPIO.setup(self.in2, GPIO.OUT)
		GPIO.setup(self.en, GPIO.OUT)
		GPIO.output(self.in1, GPIO.LOW)
		GPIO.output(self.in2, GPIO.LOW)
		self.p = GPIO.PWM(self.en, 1000)
		self.p.start(0)
		
	def setSpeed(self, speed):
		if speed == 0:
			GPIO.output(self.in1 ,GPIO.LOW)
			GPIO.output(self.in2 ,GPIO.LOW)
		elif speed > 0:
			GPIO.output(self.in1, GPIO.LOW)
			GPIO.output(self.in2, GPIO.HIGH)
			self.p.ChangeDutyCycle(speed)
		elif speed < 0:
			GPIO.output(self.in1, GPIO.HIGH)
			GPIO.output(self.in2, GPIO.LOW)
			self.p.ChangeDutyCycle(-speed)
			
			
class Cart:
	def __init__(self, m1, m2, m3, m4):
		self.fl = m1
		self.fr = m2
		self.bl = m3
		self.br = m4
		self.motors = [self.fl, self.fr, self.bl, self.br]
		
		
	def moveForward(self, speed, Time):
		for motor in self.motors:
			motor.setSpeed(speed)
		start = time.time()
		while time.time() - start < Time:
			continue
		for motor in self.motors:
			motor.setSpeed(0)		
	
	
	def rotate90(self, where):
		Time = 2 ######
		speed = 5 #####
		if where == 'left':
			self.fl.setSpeed(0)
			self.br.setSpeed(0)
			self.frsetSpeed(speed)
			self.bl.setSpeed(-speed)
		elif where == 'right':
			self.fr.setSpeed(0)
			self.bl.setSpeed(0)
			self.fl.setSpeed(speed)
			self.br.setSpeed(-speed)
		begin = time.time()
		while time.time() - begin < Time:
			continue
		for motor in self.motors:
			motor.setSpeed(0)
			

		def rotateAngle(self, angle):
			pass
	
m1 = Motor(in1_1, in1_2, en1)
m2 = Motor(in2_1, in2_2, en2)
m3 = Motor(in3_1, in3_2, en3)
m4 = Motor(in4_1, in4_2, en4)

cart = Cart(m1, m2, m3, m4)

cart.moveForward(50, 7)
#cart.rotate90('right')
#m3.setSpeed(50)
		
