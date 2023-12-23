#!/usr/bin/env python

import serial
import cv2
import time
from picamera2 import Picamera2
from keras.models import model_from_json
import numpy as np
import RPi.GPIO as GPIO          
import time
from gpiozero import LED
import struct

SERIAL_BAUD = 115200
START_FRAME = 0xABCD

serial_port = "/dev/ttyS0"

ser = serial.Serial(serial_port, SERIAL_BAUD, timeout=1)

def send_data(steer, speed):
    start_frame = struct.pack('<H', START_FRAME)
    steer_data = struct.pack('<h', steer)
    speed_data = struct.pack('<h', speed)

    start_frame_value, = struct.unpack('<H', start_frame)
    steer_value, = struct.unpack('<h', steer_data)
    speed_value, = struct.unpack('<h', speed_data)
    checksum = struct.pack('<H', (start_frame_value ^ steer_value ^ speed_value) & 0xFFFF)

    message = start_frame + steer_data + speed_data + checksum
    ser.write(message)
    
def find_intersection(x_start1, y_start1, x_end1, y_end1, x_start2, y_start2, x_end2, y_end2):
	
	if (x_end1-x_start1==0) or (x_end2-x_start2==0):
		return None, None

	m1 = (y_end1 - y_start1) / (x_end1 - x_start1)
	b1 = y_start1 - m1 * x_start1

	m2 = (y_end2 - y_start2) / (x_end2 - x_start2)
	b2 = y_start2 - m2 * x_start2
	if m1==m2:
		return None, None

	x = (b2 - b1) / (m1 - m2)
	y = m1 * x + b1

	return (x, y)

'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
''' 
    
###cap = cv2.VideoCapture(0)
        
			
class Cart:
	def __init__(self):
		self.cam = Picamera2()
		self.cam.preview_configuration.main.size = (640, 360)
		self.cam.preview_configuration.main.format = "RGB888"
		self.cam.preview_configuration.controls.FrameRate=30
		self.cam.preview_configuration.align()
		self.cam.configure("preview")
		self.cam.start()
		
		self.xcenter, self.ycenter = 0, 0
		self.angles = (1, 1)
		self.x, self.y, self.w, self.h = 220, 80, 200, 200  
		
	def image(self):	
		self.img=self.cam.capture_array()
		
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.blur = cv2.GaussianBlur(gray, (5, 5), 0)
		
		cv2.rectangle(self.img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
		
		#self.getDigit()
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
		
		edges = cv2.Canny(self.blur[self.y:self.y + self.h, self.x:self.x + self.w], 50, 150)#[self.y:self.y + self.h, self.x:self.x + self.w]
		rho = 1  # distance resolution in pixels of the Hough grid
		theta = np.pi / 180  # angular resolution in radians of the Hough grid
		threshold = 15 #15  # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 80 #80  # minimum number of pixels making up a line
		max_line_gap = 20  # maximum gap in pixels between connectable line segments
		

		lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
							min_line_length, max_line_gap)

		lengths = []
		angles = []
		if lines is not None:
			for line in lines:
				for x1,y1,x2,y2 in line:
					#cv2.line(self.img,(x1, y1),(x2, y2),(255,0,0),5)
					cv2.line(self.img,(self.w+x1+20, self.h//2-20+y1),(self.w+x2+20, self.h//2+y2-20),(255,0,0),5)
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
		
		if len(ind) < 2:
			self.angles = [1, 1]
			self.xcenter, self.ycenter = 0, 0
		else:
			angle1 = angles[ind[-1]]
			line1 = lines[ind[-1]]
			angle2 = angles[ind[-2]]   ################
			line2 = lines[ind[-2]]
			for i in range(1, len(ind)):
				if np.abs(np.abs(angles[ind[i]]) - np.abs(angle1)) >10:
					angle2 = angles[ind[i]]
					line2 = lines[ind[i]]
					break
			
			
			self.angles = [angle1, angle2]
			
			self.xcenter, self.ycenter = find_intersection(line1[0][0], line1[0][1], line1[0][2], line1[0][3],
				line2[0][0], line2[0][1], line2[0][2], line2[0][3])
			
			if (self.xcenter is not None and self.ycenter is not None and
				self.xcenter < 640 and self.ycenter < 360):
				self.img = cv2.circle(self.img, 
					(int(self.xcenter)+self.w+20, int(self.ycenter)-20+self.h//2),
					radius=10, color=(0, 0, 255), thickness=-1)
			
			
	def setSpeed(self, speedL, speedR):
		send_data(speedL, speedR)
	
	def moveForward(self, speed, Time):
		print('forward')
		start = time.time()
		while time.time() - start < Time:
			self.image()
			time.sleep(0.01)
			self.setSpeed(speed, speed)
			c = cv2.waitKey(1)
			if c == 27:
				break
		self.setSpeed(0, 0)		
	
	
	def rotate(self, where, speed, Time):
		print('rotating ', where)
		start = time.time()
		while time.time() - start < Time:
			self.image()
			c = cv2.waitKey(1)
			if c == 27:
				break
			if where == 'left':
				self.setSpeed(-speed, speed)
			elif where == 'right':
				self.setSpeed(speed, -speed)
		
		self.setSpeed(0, 0)
			
			
	def stay(self, Time):
		print('staying')
		start = time.time()
		if Time > 0:
			while time.time() - start < Time:
				self.image()
				self.setSpeed(0, 0)
				time.sleep(0.01)
				c = cv2.waitKey(1)
				if c == 27:
					break
		else:
			while True:
				self.image()
				self.setSpeed(0, 0)
				time.sleep(0.01)
				c = cv2.waitKey(1)
				if c == 27:
					break
			
			

	def rotateAngle(self):
		while True:	
			self.image()
			
			c = cv2.waitKey(1)
			if c == 27:
				break
				
			ind = np.argsort(np.abs(self.angles))
			angle = self.angles[ind[0]]
			if np.abs(angle) < 0.5:
				print('rotated')###############
				break
			
			if angle >= 0:
				self.rotate('left', 10, 0.4)
			else:
				self.rotate('right', 10, 0.4)
			
			
cart = Cart()

cart.stay(40)
cart.rotateAngle()
			
cv2.destroyAllWindows()
###cap.release()
cart.cam.stop()
