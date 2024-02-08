#!/usr/bin/env python

import serial
import cv2
import time
from picamera2 import Picamera2
from keras.models import model_from_json
import numpy as np
import time
import struct
from functools import partial

import PySimpleGUI as sg
from PIL import Image, ImageTk
from threading import Thread

SERIAL_BAUD = 115200
START_FRAME = 0xABCD
X = 640
Y = 360
ENABLE_GUI = 1

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

def enhance_red_color(image, saturation_factor=2.0):
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	h_channel, s_channel, v_channel = cv2.split(hsv_image)

	s_channel[s_channel > 60] = np.clip(s_channel[s_channel > 60] * saturation_factor, 0, 255)

	enhanced_hsv_image = cv2.merge((h_channel, s_channel, v_channel))

	enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)

	return enhanced_image

def is_point_inside_square(point, square_size, window_size):

	x, y = point
	width, height = window_size

	if x is not None and y is not None:

	    square_x1 = (width - square_size) // 2
	    square_y1 = (height - square_size) // 2

	    square_x2 = square_x1 + square_size
	    square_y2 = square_y1 + square_size

	    return square_x1 <= x <= square_x2 and square_y1 <= y <= square_y2

	else:
		return False



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

		if ENABLE_GUI == 1:
			self.create_layout()
		
		self.video_thread = Thread(target=self.image)
		self.video_thread.daemon = True  # Daemonize thread to close with the main program
		self.video_thread.start()
		
		self.cam.preview_configuration.main.size = (X, Y)
		self.cam.preview_configuration.main.format = "RGB888"
		self.cam.preview_configuration.controls.FrameRate=30
		self.cam.preview_configuration.align()
		self.cam.configure("preview")
		self.cam.start()
		
		self.xcenter, self.ycenter = 0, 0
		self.line_center = 0
		self.angles = (0, 0)
		self.angle = 0
		self.x, self.y, self.w, self.h = 220, 80, 200, 200  
		self.lines = []
		self.isMoving = 0
		self.status = 'ON'
		self.goingHome = 0
		self.pause_button = 1
		self.off_button = 1

		self.route = []
		self.routes = [[(self.stay, (4,)),
						(self.moveForward, (30,)), 
						(self.rotate, ('left', 10)), 
						(self.moveForward, (30,)),
						(self.stay, (4,))],
					   [(self.stay, (4,)), 
					    (self.moveForward, (30,)), 
					    (self.rotate, ('left', 10)),
					    (self.rotate, ('left', 10)),
					    (self.moveForward, (30,)),
					    (self.stay, (4,))]]


	def main(self):
		while self.status == 'ON':
			self.stay(0)
			self.run()
			self.stay(0)
			self.getHome()

	def create_layout(self):
		sg.theme('LightGreen4')
		self.layout = [
				[sg.Image(filename='', key='image', background_color='white')],
				#[sg.Image(filename='background.png', key='-BACKGROUND-', enable_events=True)],
				[sg.Multiline(default_text='Выберите Маршрут', size=(25, 1), disabled=True, background_color='gray', text_color='white', no_scrollbar=True),
				 sg.Button('0', size=(5, 1), font=('Helvetica', 12), button_color=('white', 'green')),
				 sg.Button('1', size=(5, 1), font=('Helvetica', 12), button_color=('white', 'green')),
				 sg.Button('BACK', size=(5, 1), font=('Helvetica', 12), button_color=('white', 'green')),
				 sg.Button('PAUSE', key='-pause-', size=(5, 1), font=('Helvetica', 12), button_color=('white', 'green'))],
				[sg.Button('TURN OFF', key = '-off-', size=(9, 1), font=('Helvetica', 12), button_color=('white', 'green')),
		         sg.Button('Exit', key='-exit-', size=(5, 1), font=('Helvetica', 12), button_color=('white', 'green'))],]

		self.window = sg.Window('cart', self.layout, resizable=True, finalize=True,
		               return_keyboard_events=True, use_default_focus=False)

		self.image_elem = self.window['image']
		self.img = None
		self.window.bind('<Escape>', '-esc-')
		self.window.bind('<Space>', '-space-')
		
	def image(self):	
		self.img=self.cam.capture_array()
		self.img = enhance_red_color(self.img)
		
		hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		#lower_red = np.array([0, 50, 50])
		#upper_red = np.array([30, 255, 255])
		lower_red = np.array([0, 20, 20])
		upper_red = np.array([10, 255, 255])
		mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
		red_lines_image = cv2.bitwise_and(self.img, self.img, mask=mask_red)
		gray_image = cv2.cvtColor(red_lines_image, cv2.COLOR_BGR2GRAY)
		self.blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
		
		cv2.rectangle(self.img, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
		
		#self.getDigit()
		self.GetAngle()
		
		if ENABLE_GUI == 1:
			self.updateGui()
		else:
			cv2.imshow("Frame", self.img)
		
	def updateGui(self):
		event, values = self.window.read(timeout=10)
			
		if event == sg.WIN_CLOSED or event == '-exit-':
			self.status = 'OFF'
			self.window.close()
			self.cam.stop()

		if event == '0':
			self.isMoving = 1
			self.route = self.routes[0]
			
		if event == '1':
			self.isMoving = 1
			self.route = self.routes[1]

		if event == '-off-' or event == '-esc-':
			if self.off_button == 1:
				self.off_button = 0
				print('turning off')
				self.status = 0
				self.window['-off-'].update(text='TURN ON')
			else:
				self.off_button = 1
				self.route = []
				print('turning on')
				self.status = 1
				self.window['-off-'].update(text='TURN OFF')
				self.main()

		if event == '-pause-' or event == ' ':
			if self.pause_button == 1:
				print('paused')
				self.pause_button = 0
				self.window['-pause-'].update(text='GO')
				self.isMoving = 0
				self.stay(0)
			if self.pause_button == 0:
				print('continue')
				self.pause_button = 1
				self.window['-pause-'].update(text='PAUSE')
				self.isMoving = 1


		if event == 'BACK':
			self.isMoving = 1
			self.getHome()

		if self.img is not None:
		    img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
		    img_tk = ImageTk.PhotoImage(img)

		    # Update the GUI window
		    self.image_elem.update(data=img_tk)

		# Schedule the next GUI update
		self.window.refresh()

	def getHome(self):
		print('returning')
		self.goingHome = 1
		self.rotate('right', 10)
		self.rotate('right', 10)
		for func, params in self.route[::-1]:
			partial_func = partial(func, *params)
			partial_func()
		self.rotate('right', 10)
		self.rotate('right', 10)
		self.isMoving = 0
		print('at home')
		self.goingHome = 0

	def run(self):
		if self.isMoving == 1:
			print('enter run')
			for func, params in self.route:
				partial_func = partial(func, *params)
				partial_func()
			self.isMoving = 0
		print('exit run')
		
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
		threshold = 30 #15  # minimum number of votes (intersections in Hough grid cell)
		min_line_length = 30 #80  # minimum number of pixels making up a line
		max_line_gap = 5  # maximum gap in pixels between connectable line segments
		

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
		
		if len(ind)>0:
			self.line_center = (lines[ind[-1]][0][0] + lines[ind[-1]][0][2])//2
			self.angle = angles[ind[-1]]

		if len(ind) < 2:
			#self.angles = [1, 1]
			self.xcenter, self.ycenter = -10, -10
		else:
			angle1 = angles[ind[-1]]
			line1 = lines[ind[-1]]
			for i in range(1, len(ind)):
				if np.abs(np.abs(angles[ind[i]]) - np.abs(angle1)) >10:
					angle2 = angles[ind[i]]
					line2 = lines[ind[i]]
					self.lines = [line1, line2]
					self.angles = [angle1, angle2]
					self.xcenter, self.ycenter = find_intersection(line1[0][0], line1[0][1], line1[0][2], line1[0][3],
						line2[0][0], line2[0][1], line2[0][2], line2[0][3])
					break
			
	
			if is_point_inside_square((self.xcenter, self.ycenter), 200, (self.w, self.h)):
				self.img = cv2.circle(self.img, 
					(int(self.xcenter)+self.w+20, int(self.ycenter)-20+self.h//2),
					radius=10, color=(0, 0, 255), thickness=-1)
			
			
	def setSpeed(self, speedL, speedR):
		send_data(speedL, speedR)
	
	def get_speed(self, speed):
		kp_angle = 0.17
		kp_center = 0.08
		self.speed = speed
		
		center = self.line_center - self.w // 2
		
		lspeed = int(speed + kp_angle*self.angle - kp_center*center)
		rspeed = int(speed - kp_angle*self.angle + kp_center*center)
	
		return lspeed, rspeed
	
	def moveForward(self, speed):
		print('forward')
		start = time.time()
		while self.isMoving == 1:
			self.image()
			c = cv2.waitKey(1)
			time.sleep(0.01)
			self.setSpeed(*self.get_speed(speed))
			self.speed = 0
			if (is_point_inside_square((self.xcenter, self.ycenter), 120, (self.w, self.h)) and
				time.time() - start > 5):
				print('choose path')
				break
			if c == 27:
				break
		self.setSpeed(0, 0)		
	
	
	def rotate(self, where, speed):
		if self.goingHome == 1:
			speed = -speed
		print('rotating ', where)
		start = time.time()	
		while self.isMoving == 1:
			self.image()
			print(self.angles)
			c = cv2.waitKey(1)
			if c == 27:
				break
			if where == 'left':
				self.setSpeed(-speed, speed)
				if (np.abs(self.angles[0])<10 and 
				(np.abs(self.angles[1]+90)<5 or np.abs(self.angles[1]-90)<5) and
				 time.time()-start>0.7):
					self.setSpeed(0, 0)
					break
			elif where == 'right':
				self.setSpeed(speed, -speed)
				if (np.abs(self.angles[0])<10 and 
				(np.abs(self.angles[1]+90)<5 or np.abs(self.angles[1]-90)<5) and
				 time.time()-start>0.7):
					self.setSpeed(0, 0)
					break
			
		self.setSpeed(0, 0)


	def rotate_test(self, speed):
		while True:
			if len(self.angles) > 1:
				if self.lines[1][0] + self.lines[1][2] < self.lines[0][0] + self.lines[0][2]:
					where = 'left'
				else:
					where = 'right'
				temp_angle = np.abs(self.angles[1])
				break
		print('rotating ', where)
		start = time.time()	
		while True:
			self.image()
			c = cv2.waitKey(1)
			if c == 27:
				break
			if where == 'left':
				self.setSpeed(-speed, speed)
				if (np.abs(self.angles[0])<5 and 
				np.abs(np.abs(self.angles[1]) - temp_angle) < 5 and
				 time.time()-start>0.4):
					self.setSpeed(0, 0)
					break
			elif where == 'right':
				self.setSpeed(speed, -speed)
				if (np.abs(self.angles[0])<5 and 
				np.abs(np.abs(self.angles[1]) - temp_angle) < 5 and
				 time.time()-start>0.4):
					self.setSpeed(0, 0)
					break
			
		self.setSpeed(0, 0)
			
			
	def stay(self, Time):
		print('staying')
		start = time.time()
		if Time > 0:
			while time.time() - start < Time and self.isMoving == 0:
				print(self.angle)
				self.image()
				self.setSpeed(0, 0)
				time.sleep(0.01)
				c = cv2.waitKey(1)
				if c == 27:
					break
		else:
			while self.isMoving == 0:
				self.image()
				self.setSpeed(0, 0)
				time.sleep(0.01)
				c = cv2.waitKey(1)
				if c == 27:
					break
			print(start)
			
			
	
cart = Cart()
cart.main()

cv2.destroyAllWindows()
###cap.release()
cart.cam.stop()
