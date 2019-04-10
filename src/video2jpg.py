import cv2
import numpy
import os

def v2j(videopath, image_save_path):
    vc = cv2.VideoCapture(videopath)
    c = 1
    if vc.isOpened():
        rval,frame = vc.read()
    else:
        rval = False
    while rval:
        rval,frame = vc.read()
        cv2.imwrite(image_save_path+str(c+4000)+'.jpg',frame)
        c = c + 1
        cv2.waitKey(1)
    print('Video2JPG Completed')
    vc.release()

def j2v(imagepath, video_save_path):
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video_writer = cv2.VideoWriter(filename=video_save_path,fourcc=fourcc,fps=fps,frameSize=(1000,600))
    im_name = os.listdir(imagepath)
    for i in range(len(im_name)):
        frame = cv2.imread(imagepath+str(i)+'.jpg')
        video_writer.write(frame)
    print('JPG2Video Completed')
    video_writer.release()
