import cv2
import time
from picamera2 import Picamera2
from keras.models import model_from_json
import numpy as np
import RPi.GPIO as GPIO          
import time
from gpiozero import LED

en1 = 19
en2 = 12
en3 = 18
en4 = 13
in1_2 = 26
in1_1 = 6
in2_2 = 5
in2_1 = 0
in3_1 = 20
in3_2 = 16
in4_1 = 1
in4_2 = 7


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
###cap = cv2.VideoCapture(0)
        
    
class Motor:
	def __init__(self, in1, in2, en):
		self.in1 = in1
		self.in2 = in2
		self.en = en
		GPIO.setup(self.in1, GPIO.OUT)
		GPIO.setup(self.in2, GPIO.OUT)
		GPIO.output(self.in1, GPIO.LOW)
		GPIO.output(self.in2, GPIO.LOW)
		self.p = LED(en)
		self.p.value = 0
		
	def setSpeed(self, speed):
		if speed == 0:
			GPIO.output(self.in1 ,GPIO.LOW)
			GPIO.output(self.in2 ,GPIO.LOW)
		elif speed > 0:
			GPIO.output(self.in1, GPIO.LOW)
			GPIO.output(self.in2, GPIO.HIGH)
			self.p.value = speed
		elif speed < 0:
			GPIO.output(self.in1, GPIO.HIGH)
			GPIO.output(self.in2, GPIO.LOW)
			self.p.value = -speed
			
			
class Cart:
	def __init__(self, m1, m2, m3, m4):
		self.fl = m1
		self.fr = m2
		self.bl = m3
		self.br = m4
		self.motors = [self.fl, self.fr, self.bl, self.br]
		
		self.cam = Picamera2()
		self.cam.preview_configuration.main.size = (640, 360)
		self.cam.preview_configuration.main.format = "RGB888"
		self.cam.preview_configuration.controls.FrameRate=30
		self.cam.preview_configuration.align()
		self.cam.configure("preview")
		self.cam.start()
		
		self.angles = (1, 1)
		self.x, self.y, self.w, self.h = 0, 0, 300, 300 
		
	def image(self):	
		self.img=self.cam.capture_array()
		
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.blur = cv2.GaussianBlur(gray, (5, 5), 0)
		
		cv2.rectangle(self.img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
		
		self.getDigit()
		self.GetAngle()
		
		cv2.imshow("Frame", self.img)
		
		
	def getDigit(self):
		ret, thresh = cv2.threshold(self.blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		newImage = thresh[self.y:self.y + self.h, self.x:self.x + self.w] 
		newImage = cv2.resize(newImage, (28, 28))
		newImage = np.array(newImage)
		newImage = newImage.astype('float32')
		newImage /= 255
		newImage = newImage.reshape(28, 28, 1)
		newImage = np.expand_dims(newImage, axis=0)
		ans = ''
		ans = model.predict(newImage, verbose=0).argmax()
		
		cv2.putText(self.img, "CNN : " + str(ans), (10, 320),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						
		self.digit = ans 
    
	def GetAngle(self):
		edges = cv2.Canny(self.blur, 50, 150)[self.y:self.y + self.h, self.x:self.x + self.w]
		rho = 1  # distance resolution in pixels of the Hough grid
		theta = np.pi / 180  # angular resolution in radians of the Hough grid
		threshold = 15  # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 80  # minimum number of pixels making up a line
		max_line_gap = 20  # maximum gap in pixels between connectable line segments

		lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
							min_line_length, max_line_gap)

		lengths = []
		angles = []
		if lines is not None:
			for line in lines:
				for x1,y1,x2,y2 in line:
					cv2.line(self.img,(x1,y1),(x2,y2),(255,0,0),5)
					
					lengths.append(np.sqrt((x1-x1)**2+(y1-y2)**2))
					if y1>y2:                                           ########test later
						dx = x1-x2
						angles.append(180/np.pi*np.arctan(dx/np.abs(y1 - y2)))
					elif y1 < y2:
						dx = x2-x1
						angles.append(180/np.pi*np.arctan(dx/np.abs(y1 - y2)))
					elif y1 == y2:
						angles.append(90)
						
		ind = np.argsort(lengths)
		print(angles)
		if len(ind) < 2:
			self.angles = [1, 1]
		else:
			angle1 = angles[ind[-1]]
			angle2 = angles[ind[-2]]   ################
			for i in range(1, len(ind)):
				if np.abs(np.abs(angles[ind[i]]) - np.abs(angle1)) >10:
					angle2 = angles[ind[i]]
					break
			
			self.angles = [angle1, angle2]
		
	
	def moveForward(self, speed, Time):
		start = time.time()
		for motor in self.motors:
			motor.setSpeed(speed)
		while time.time() - start < Time:
			self.image()
			c = cv2.waitKey(1)
			if c == 27:
				break
		for motor in self.motors:
			motor.setSpeed(0)		
	
	
	def rotate(self, where, speed, Time):
		start = time.time()
		if where == 'left':
			self.fl.setSpeed(0)
			self.br.setSpeed(0)
			self.fr.setSpeed(speed)
			self.bl.setSpeed(-speed)
		elif where == 'right':
			self.fr.setSpeed(0)
			self.bl.setSpeed(0)
			self.fl.setSpeed(speed)
			self.br.setSpeed(-speed)
		while time.time() - start < Time:
			self.image()
			c = cv2.waitKey(1)
			if c == 27:
				break
		for motor in self.motors:
			motor.setSpeed(0)
			
			
	def stay(self, Time):
		print('staying')
		start = time.time()
		for motor in self.motors:
			motor.setSpeed(0)
		if Time > 0:
			while time.time() - start < Time:
				print(self.angles)
				self.image()
				c = cv2.waitKey(1)
				if c == 27:
					break
		else:
			while True:
				self.image()
				c = cv2.waitKey(1)
				if c == 27:
					break
			
			

	def rotateAngle(self):
		while True:	
			self.image()
			
			c = cv2.waitKey(1)
			if c == 27:
				break
				
			print(self.angles)
			ind = np.argsort(np.abs(self.angles))
			angle = self.angles[ind[0]]
			if np.abs(angle) < 0.5:
				print('rotated')###############
				break
			
			if angle >= 0:
				self.rotate('left', 0.02, 0.005)
			else:
				self.rotate('right', 0.1, 0.01)
			
			

	
m1 = Motor(in1_1, in1_2, en1)
m2 = Motor(in2_1, in2_2, en2)
m3 = Motor(in3_1, in3_2, en3)
m4 = Motor(in4_1, in4_2, en4)

cart = Cart(m1, m2, m4, m3)

cart.stay(5)
cart.rotateAngle()
cart.stay(0)

#cart.moveForward(0.1, 5)
#time.sleep(10)
#cart.stay(30)
#cart.moveForward(0.05, 1)
#cart.rotate('right', 0.1, 0.58)       ###############
#cart.stay(2)
#cart.moveForward(0.05, 0.5)
#cart.rotate('left', 0.1, 0.78)         #############

#cart.br.setSpeed(0.1)
#time.sleep(1)
#cart.br.setSpeed(-0.1)
#time.sleep(1)
#cart.br.setSpeed(0)			

cv2.destroyAllWindows()
###cap.release()
cart.cam.stop()
