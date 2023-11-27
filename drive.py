import RPi.GPIO as GPIO          
import time

en1 = 26
en2 = 22
en3 = _
en4 = _
in1_1 = 19
in1_2 = 13
in2_1 = 6
in2_2 = 5
in3_1 = 6
in3_2 = 5
in4_1 = 6
in4_2 = 5

orient = 1

GPIO.setmode(GPIO.BCM)
GPIO.setup(in1_1,GPIO.OUT)
GPIO.setup(in1_2,GPIO.OUT)
GPIO.setup(in2_1,GPIO.OUT)
GPIO.setup(in2_2,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.setup(en3,GPIO.OUT)
GPIO.setup(en4,GPIO.OUT)
GPIO.output(in1_1,GPIO.LOW)
GPIO.output(in1_2,GPIO.LOW)
GPIO.output(in2_1,GPIO.LOW)
GPIO.output(in2_2,GPIO.LOW)
GPIO.output(in3_1,GPIO.LOW)
GPIO.output(in3_2,GPIO.LOW)
GPIO.output(in4_1,GPIO.LOW)
GPIO.output(in4_2,GPIO.LOW)
p1=GPIO.PWM(en1,1000)
p2=GPIO.PWM(en2,1000)
p1=GPIO.PWM(en3,1000)
p2=GPIO.PWM(en4,1000)
p1.start(0)
p2.start(0)
p3.start(0)
p4.start(0)

motors = [(in1_1, in1_2, en1), (in2_1, in2_2, en2), (in3_1, in3_2, en3), (in4_1, in4_2, en4)]

def setSpeed(motor, speed):
	if speed == 0:
		GPIO.output(motor[0] ,GPIO.LOW)
		GPIO.output(motor[1] ,GPIO.LOW)
	elif speed > 0:
		GPIO.output(motor[0] ,GPIO.HIGH)
		GPIO.output(motor[1] ,GPIO.LOW)
		motor[3].ChangeDutyCycle(speed)
	else:
		GPIO.output(motor[0] ,GPIO.LOW)
		GPIO.output(motor[1] ,GPIO.HIGH)
		motor[3].ChangeDutyCycle(speed)
		

def moveForward(speed, time, motors):
	start = time.time()
	for motor in motors:
			setSpeed(motor, speed)
	while time.time() < time:
		continue
	for motor in motors:
			setSpeed(motor, 0)
			
			
	
	
def rotate90(where, motors):
	time = 0 ######
	speed = 0 #####
	begin = time.time()
	while time.time() < begin:
		if where == 'left':
			setSpeed(motors[0], 0)
			setSpeed(motors[2], 0)
			setSpeed(motors[1], speed)
			setSpeed(motors[3], -speed)
		elif where == 'right':
			setSpeed(motors[1], 0)
			setSpeed(motors[2], 0)
			setSpeed(motors[0], speed)
			setSpeed(motors[3], -speed)
	for motor in motors:
		setSpeed(motor, 0)


def rotateAngle(angle, motors)
	
		
