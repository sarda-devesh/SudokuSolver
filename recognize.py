import numpy as np
import cv2
import os
import imutils
from keras.models import load_model, model_from_json

def load_entire_model(path_to_model):
	json_file_path = os.path.join(path_to_model, 'model.json')
	json_file = open(json_file_path, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	h5_file_path = os.path.join(path_to_model, 'model.h5')
	loaded_model.load_weights(h5_file_path)
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return loaded_model

def perform_recognition(path_to_image, path_to_model):
	model = load_entire_model(path_to_model)

	image = cv2.imread(path_to_image)

	'''
	x_pixel = 10
	y_pixel = 1
	blue = int(image[x_pixel, y_pixel, 0])
	green = int(image[x_pixel, y_pixel, 1])
	red = int(image[x_pixel, y_pixel, 2])
	print((blue, green, red))
	'''

	image = imutils.resize(image, width= 320)

	print(image.shape)


	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
	_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	thresh = cv2.dilate(thresh,None) 

	_,cnts,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])

	digits = []
	boxes = []

	for i,c in enumerate(cnts):
		if cv2.contourArea(c)<avgCntArea/10:
			continue
		mask = np.zeros(gray.shape,dtype="uint8") 
		(x,y,w,h) = cv2.boundingRect(c)
		hull = cv2.convexHull(c)
		cv2.drawContours(mask,[hull],-1,255,-1) 
		mask = cv2.bitwise_and(thresh,thresh,mask=mask)
		digit = mask[y-8:y+h+8,x-8:x+w+8] 
		if(len(digit) == 0):
			continue
		digit = cv2.resize(digit,(28,28))
		boxes.append((x,y,w,h))
		digits.append(digit)

	digits = np.array(digits)
	
	digits = digits.reshape(digits.shape[0],28,28,1)
	labels = model.predict_classes(digits)

	cv2.imshow("Original",image)
	cv2.imshow("Thresh",thresh)
	
	for (x,y,w,h),label in sorted(zip(boxes,labels)):
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
		cv2.putText(image,str(label),(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
	
	cv2.imshow("Recognized",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def scale_image(path_to_image, factor):
	image = cv2.imread(path_to_image + ".png")
	print(image.shape)
	image = cv2.resize(image, None, fx = factor, fy = factor)
	name = path_to_image + "_" + str(factor) + ".png"
	cv2.imwrite(name, image)

if __name__ == '__main__':
	path_to_image = os.path.join('images','Cropped.png')
	#scale_image(path_to_image, 10)
	path_to_model = "output"
	perform_recognition(path_to_image, path_to_model)