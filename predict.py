# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:00:26 2022

@author: atver
"""

import PIL
import cv2 as cv
import numpy as np
from Lib.run_model import*
from time import sleep as sleep
import bbox_visualizer as bbv
import pandas as pd
import os

save = False
# get images from folder
folder = 'image_for_predict'
filenames = os.listdir(folder)

for img_fn in filenames:  
    full_path = os.path.join(folder, img_fn)
    image = cv.imread(full_path)
    fr_w = image.shape[1]
    fr_h = image.shape[0]
    # Make predictions
    image_pil = PIL.Image.fromarray(image)
    mr = model_predict(image_pil) # model results # a pd.DataFrame with columns 'xmin','ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
    for ind in range(len(mr.index)):
        bbox = [mr.iloc[ind,0], mr.iloc[ind,1], mr.iloc[ind,2], mr.iloc[ind,3]]
        bbox = [int(x) for x in bbox]
        image = bbv.draw_rectangle(image, bbox)
        image = bbv.add_label(image, str(mr.iloc[ind,6])+': '+str(round(mr.iloc[ind,4],4)), bbox)
    title = f"{img_fn}" 
    font, fontScale, fontColor, thick = (cv.FONT_HERSHEY_SIMPLEX,1,[255,0,0],4)
    cv.putText(frame, title, (int(fr_w / 50), int(fr_h / 15)), font, fontScale, fontColor, thick)
    cv.imshow('Frame', image)
    # RUN ONCE W/O WAITKEY! then again in same console with wait key (work around for bug) 
    cv.waitKey(0)
    if save:
        img_fn += "_with_pred"
        cv.imwrite(img_fn, image)
              
# Closes all the frames
cv.destroyAllWindows()















