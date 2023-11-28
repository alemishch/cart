import cv2
import time
from picamera2 import Picamera2
from keras.models import model_from_json
import numpy as np

cam = Picamera2()
fps = 30
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate=30
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
###cap = cv2.VideoCapture(0)

    
def getDigit(img_processed, img_raw, x, y, h, w):
    ret, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    newImage = thresh[y:y + h, x:x + w] #saving threshold image into new image
    #resizing the image into 28*28. using opencv to resize image without distortion. 
    newImage = cv2.resize(newImage, (28, 28))
    # converting images to numpy array of images
    newImage = np.array(newImage)
    newImage = newImage.astype('float32')
    # normalizing the image 
    newImage /= 255
    # reshaping numpy array of images. numpy.reshape just change the shape attribute without changing the data (images) at all.
    newImage = newImage.reshape(28, 28, 1)
    # inserting a new axis at the 0th location to expand the size of numpy array of images of 4 from (28,28,1) to (1,28,28,1)
    newImage = np.expand_dims(newImage, axis=0)
    ans = ''
    
    # predicting the model    
    ans = model.predict(newImage, verbose=0).argmax()
    
    #putText(img, text, org(it is a point representing the bottom left corner text string in the image), fontFace, fontScale, Scalar color, int thickness)
    cv2.putText(img, "CNN : " + str(ans), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
    return ans    
    
def GetAngle(img_processed, img_raw, x, y, h, w):
    edges = cv2.Canny(img_processed, 50, 150)[y:y + h, x:x + w]
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
                cv2.line(img_raw,(x1,y1),(x2,y2),(255,0,0),5)
                lengths.append(np.sqrt((x1-x1)**2+(y1-y2)**2))
                if y1>y2:                                           ########test later
                    dx = x1-x2
                else:
                    dx = x2-x1
                angles.append(180/np.pi*np.arctan(dx/np.abs(y1 - y2))) ############# test zero
    ind = np.argsort(lengths)
    
    if len(ind) < 2:
        return 0, 0
    else:
        angle1 = angles[ind[0]]
        angle2 = angles[ind[1]]
        for i in range(1, len(ind)):
            if np.abs(np.abs(angles[ind[i]]) - np.abs(angle1)) >10:
                angle2 = angles[ind[i]]
        
        return angle1, angle2
    

    
while True:
    ###ret, img = cap.read()
    
    tStart=time.time()

    img=cam.capture_array()
    
    x, y, w, h = 0, 0, 300, 300 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    num = getDigit(blur, img, x, y, h, w)
    angles = GetAngle(blur, img, x, y, h, w)
    print(angles)
    
    cv2.imshow("Frame", img)
    
    c = cv2.waitKey(1)
    if c == 27:
        break
        
    tEnd=time.time()
    # time between two epochs
    looptime=tEnd-tStart
    fps=1/looptime
    
    
cv2.destroyAllWindows()
###cap.release()
cam.stop()
