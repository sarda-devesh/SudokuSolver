import cv2
import numpy as np
import imutils
import argparse
from imutils.perspective import four_point_transform, order_points
from skimage.filters import threshold_adaptive
from skimage.segmentation import clear_border
from keras.models import load_model
from sudoku import SolveSudoku
import os 
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

def solve_sudoku(path_to_image,path_to_model):
    image = cv2.imread(path_to_image)
    poly = None 
    image = imutils.resize(image,width=800)
    #cv2.imshow("Original",image)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    thresh = threshold_adaptive(blurred,block_size=5,offset=1).astype("uint8")*255
    cv2.imshow("Thresholded",thresh)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'): 
       return None 
    _,cnts,_ = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    mask = np.zeros(thresh.shape,dtype="uint8")
    c = cnts[1]
    clone = image.copy()
    peri = cv2.arcLength(c,closed=True)
    poly = cv2.approxPolyDP(c,epsilon=0.02*peri,closed=True)
    if len(poly) == 4:
        print(poly)
        cv2.drawContours(thresh,[poly],-1,(0,255,0),2)
        cv2.imshow("Contours",thresh)
        warped = four_point_transform(image,poly.reshape(-1,2))
        cv2.imshow("Warped",warped)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'): 
        return None
    warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    winX = int(warped.shape[1]/8.7)
    winY = int(warped.shape[0]/8.7)
    x_ratio = warped.shape[1]/winX
    y_ratio = warped.shape[0]/winX
    model = load_entire_model(path_to_model)
    labels = []
    centers = []
    for y in range(0,warped.shape[0],winY):
        for x in range(0,warped.shape[1],winX):
            window = warped[y:y+winY,x:x+winX]
            clone = warped.copy()
            digit = cv2.resize(window,(28,28))
            _,digit = cv2.threshold(digit,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            digit = clear_border(digit)
            copy = cv2.copyMakeBorder(digit, 0, 0, 0, 200, cv2.BORDER_CONSTANT, value= (0, 0, 0))
            cv2.imshow("Digit",copy)
            numPixels = cv2.countNonZero(digit)
            if numPixels<5:
                label = 0
            else:
                label = model.predict_classes([digit.reshape(1,28,28,1)])[0]
            labels.append(label)
            centers.append(((x+x+winX)//2,(y+y+winY+6)//2))
            cv2.rectangle(clone,(x,y),(x+winX,y+winY),(0,0,255),2)
            cv2.imshow("Window",clone)
            cv2.waitKey(0)
    temp = np.array(labels)
    grid = np.reshape(temp, (9, 9))
    print("Got grid")
    gz_indices = zip(*np.where(grid==0))
    gz_centers = np.array(centers).reshape(9,9,2)
    sudoku = SolveSudoku(labels)
    grid = sudoku.solve()
    con = 5
    for row,col in gz_indices: 
        center_x, center_y = gz_centers[row][col]
        cv2.putText(warped,str(grid[row][col]), (center_x, center_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
    cv2.imshow("Solved",warped)
    cv2.waitKey(0)
    pt_src = [[0,0],[warped.shape[1],0],[warped.shape[1],warped.shape[0]],[0,warped.shape[0]]]
    pt_src = np.array(pt_src,dtype="float")
    pt_dst = poly.reshape(4,2)
    pt_dst = pt_dst.astype("float")
    pt_src = order_points(pt_src)
    pt_dst = order_points(pt_dst)
    H,_ = cv2.findHomography(pt_src,pt_dst)
    im_out = cv2.warpPerspective(warped,H,dsize=(gray.shape[1],gray.shape[0]))
    im_out = cv2.addWeighted(gray,0.9,im_out,0.2,0)
    cv2.imshow("Projected",im_out)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def solve_sudoku_with_video(path_to_model): 
    video = cv2.videoCapture(0)
    end_frame = None
    live = True
    while(end_frame != None):
        ret,image = video.read()
        if(ret == False):
            print("Can't get input")
            break
        if(live):
            cv2.imshow("Display",image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'): 
            live = False 
        elif key == ord('a'): 
            live = True 
        elif key == ord('s'):
            end_frame = image 
    frame_name = os.path.join("images", "Sudoku_frame.jpg")
    cv2.imwrite(frame_name, end_frame) 
    solve_sudoku(frame_name,path_to_model)

if __name__ == "__main__":
    path_to_image = os.path.join("images", "input.jpeg")
    path_to_model = "model"
    solve_sudoku(path_to_image, path_to_model)