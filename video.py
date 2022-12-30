# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:59:31 2022

@author: atver

Written to run on the VM
"""

import PIL
import cv2 as cv
import numpy as np
import datetime
from Lib.run_model import*
import bbox_visualizer as bbv
import pandas as pd
import pickle5 as pickle


def get_useLRH(use_path,use_fn,frNum,LR):
    LR = "tools_right/" if LR else "tools_left/"
    with open(use_path + LR + use_fn, 'r') as uf:
        for ln in uf:
            sf, ff, useH = ln.strip().split()
            if int(sf) <= int(frNum) <= int(ff):
                return useH

videos = ["/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P022_balloon1.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P023_tissue2.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P024_balloon1.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P025_tissue2.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P026_tissue1.wmv"]

vid = videos[0]
# or loop over the videos
for vid in videos:
    vidm = vid.split('/')[-1][:4]
    cap = cv.VideoCapture(vid)
    nFr = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Check if camera/vid opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        quit()
        
    frNum = 1
    # Define our vid maker
    fr_w = int(cap.get(3))
    fr_h = int(cap.get(4))
    time_str = str(datetime.datetime.now()).split('.')[0].replace(':', ' ').replace('-', ' ').replace(' ', '_')
    # Dataframe to collect the predictions
    seg_preds = pd.DataFrame(columns = ["LHGT","RHGT","LH straight","RH straight", "LH fp","RH fp","LH smoothM15", "RH smoothM15","LH smoothM35", "RH smoothM35","LH smoothM65", "RH smoothM65"])
    all_bboxes = pd.DataFrame()
    # Read until video is completed
    while (cap.isOpened()) and frNum < 100000:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Pass the frame into the chosen trained network here as PIL
            image_pil = PIL.Image.fromarray(frame)
            mr = model_predict(image_pil) # model results # a pd.DataFrame with columns 'xmin','ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
            bboxes = mr
            bboxes["Frame"] = frNum
            all_bboxes = pd.concat([all_bboxes,bboxes], axis=0)
            
            # # File of all bbox predicitons from model
            # with open('/home/student/Desktop/hW1/Surgical-OD-HW1/apply2vid' + time_str+ '.txt', 'a') as log:
            #     log.write(f"Frame # {frNum}"+"\n" + str(mr)+"\n")
            #     log.write("type: "  + str(type(mr))+"\n\n")
                
            # get ground truth usages
            use_path = "/home/student/Desktop/hW1/Surgical-OD-HW1/HW1_dataset/tool_usage/" 
            use_fn = vid.split('/')[-1].replace('.wmv','.txt')
            useLH = get_useLRH(use_path,use_fn,frNum,0)
            seg_preds.loc[frNum,"LHGT"] = useLH
            useRH = get_useLRH(use_path,use_fn,frNum,1)
            seg_preds.loc[frNum,"RHGT"] = useRH
            
            
            # split predicted bboxes by class
            mr_LH = mr.loc[(mr["class"]==1) | (mr["class"]==3) | (mr["class"]==5) | (mr["class"]==7)]
            mr_RH = mr.loc[(mr["class"]==0) | (mr["class"]==2) | (mr["class"]==4) | (mr["class"]==6)]
            # store the orig predicted classes per hand 
            seg_preds.loc[frNum,"LH straight"] = np.array([LHc for LHc in mr_LH["class"]]) # first pass, vs smooothed
            seg_preds.loc[frNum,"RH straight"] = np.array([RHc for RHc in mr_RH["class"]])
            
            # Begin a first pass of smoothing: deal with cases of no pred for that hand or multiple
            # If a prediction exists for that hand choose the prediction that has greatest confidence
            fp_LH = mr_LH.sort_values('confidence', ascending = 1)
            fp_RH = mr_RH.sort_values('confidence', ascending = 1)
            if not fp_LH.empty:
                fp_LH = fp_LH.iloc[0,:]
            else:
                # get a close RH bbox to last LH with better confidence or assign same as previous
                try:
                    mr_RH['d prev RH'] = np.sqrt((prevRH.loc['xmin'] - mr_RH.iloc[:,0])**2 +
                                             (prevRH.loc['ymin'] - mr_RH.iloc[:,1])**2)
                    mr_RH['d prev LH'] = np.sqrt((prevLH.loc['xmin'] - mr_RH.iloc[:,0])**2 +
                                             (prevLH.loc['ymin'] - mr_RH.iloc[:,1])**2)
                    fp_LH = mr_RH.loc[mr_RH['d prev LH']<20]
                    fp_LH = fp_LH.sort_values('confidence', ascending = 1)  
                    fp_LH.loc[0,'class'] = fp_LH.loc[0,'class'] + 1
                    fp_LH = fp_LH.iloc[0,:]
                except NameError: # no previous 
                    # get furthest to right (their left) or default close to the side of screen 
                    if len(mr_RH.index)>1:
                        fp_LH = mr_RH.nlargest(1,'xmin').reset_index() # largest
                        fp_LH.loc[0,'class'] = fp_LH.loc[0,'class'] + 1
                        fp_LH = fp_LH.iloc[0,:]
                    elif len(mr_RH.index)<=1:
                        fp_LH = fp_LH.loc[0,'xmin'] = fr_w/4
                        fp_LH = fp_LH.loc[0,'ymin'] = fr_h/2
                        fp_LH = fp_LH.loc[0,'class'] = 7
                        fp_LH = fp_LH.iloc[0,:]
                except KeyError: # no close bbox but previous exists set as previous
                    fp_LH = prevLH
                    
            if not fp_RH.empty:
                fp_RH = fp_RH.iloc[0,:]
                assert(type(fp_RH) == pd.core.series.Series) 
            else:
                # get a close LH bbox to last RH with better confidence or assign same as previous
                try:
                    mr_LH['d prev RH'] = np.sqrt((prevRH.loc['xmin'] - mr_LH.iloc[:,0])**2 +
                                                 (prevRH.loc['ymin'] - mr_LH.iloc[:,1])**2)
                    mr_LH['d prev LH'] = np.sqrt((prevLH.loc['xmin'] - mr_LH.iloc[:,0])**2 +
                                                 (prevLH.loc['ymin'] - mr_LH.iloc[:,1])**2)
                    fp_RH = mr_LH.loc[mr_LH['d prev RH']<20]
                    fp_RH = fp_RH.sort_values('confidence', ascending = 1)  
                    fp_RH.loc[0,'class'] = fp_RH.loc[0,'class'] - 1
                    fp_RH = fp_RH.iloc[0,:]
                    assert(type(fp_RH) == pd.core.series.Series)  
                except NameError: # no previous 
                    # get furthest to left (their right) or default close to the side of screen 
                    if len(mr_LH.index)>1:
                        fp_RH = mr_LH.nsmallest(1,'xmin').reset_index() # smallest
                        fp_RH.loc[0,'class'] = fp_RH.loc[0,'class'] - 1
                        fp_RH = fp_RH.iloc[0,:]
                        assert(type(fp_RH) == pd.core.series.Series)  
                    elif len(mr_LH.index)<=1:
                        fp_RH = fp_RH.loc[0,'xmin'] = fr_w *3/4
                        fp_RH = fp_RH.loc[0,'ymin'] = fr_h/2
                        fp_RH = fp_RH.loc[0,'class'] = 6
                        fp_RH = fp_RH.iloc[0,:]
                        assert(type(fp_RH) == pd.core.series.Series)  
                except KeyError: # no close bbox but previous exists set as previous
                    fp_RH = prevRH
                    assert(type(fp_RH) == pd.core.series.Series)  
        
            prevRH = fp_RH # should be a series
            prevLH = fp_LH # should be a series
            
            seg_preds.loc[frNum,"LH fp"] = fp_LH['class']
            seg_preds.loc[frNum,"RH fp"] = fp_RH['class']
            
            # write this next line of the seg_pred panda to file 
            with open('/home/student/Desktop/hW1/Surgical-OD-HW1/' + vidm+ 'a2v_modelsegpred' + time_str +'.txt', 'a') as msp:
                if frNum ==1:
                    for col in [str(col) for col in seg_preds.columns]:
                        msp.write('{:<15}|'.format(col))
                    msp.write('\n')
                    # msp.write('{:<20}'.format('LHuse pred')+'|' + '{:<20}'.format('RHuse pred')+ "\n")
                msp.write(str(seg_preds.loc[frNum].values) + '\n')
                    
            if (frNum%int(nFr/5))==0:
                print(f"processed {frNum}/{nFr} frames")
            # increment the frame number
            frNum += 1
        # Break the loop
        else:
            break
    
    # print("type of all_bboxes", type(all_bboxes))
    all_bboxes.to_csv(vidm +'_all_bboxes.csv')
    # with open(vidm+'_all_bboxes.txt', 'wb') as f:
    #     pickle.dump([all_bboxes],f) 
        
    # with open(vidm+'all_bboxes.txt', 'rb') as f:
    #     all_bboxes = pickle.load(f) 
    # print("after reloading", type(all_bboxes)) # why is it coming back as a list instead of a panda??!!!!
    
    # with open(vidm+'last_seg_preds.txt', 'wb') as f:
    #     pickle.dump([vid, seg_preds, nFr],f) 
    seg_preds.to_csv(vidm +'_seg_preds.csv')
        
    # with open(vidm+'last_seg_preds.txt', 'rb') as f:
    #     vid, seg_preds, nFr = pickle.load(f) 
        
    # Smoothing the predicted segmentations after the first pass. Majority voting
    for ind in range(1,1+len(seg_preds.index)):
        s_ind = ind - 7 if ind>=8 else 1
        e_ind = ind + 7 if ind<=nFr-7 else nFr
        seg_preds.loc[ind,"LH smoothM15"] = seg_preds.loc[s_ind:e_ind]['LH fp'].mode().iloc[0] # scalar. if tie picks lower value
        seg_preds.loc[ind,"RH smoothM15"] = seg_preds.loc[s_ind:e_ind]['RH fp'].mode().iloc[0]
    
        s_ind = ind - 17 if ind>=18 else 1
        e_ind = ind + 17 if ind<=nFr-17 else nFr
        seg_preds.loc[ind,"LH smoothM35"] = seg_preds.loc[s_ind:e_ind]['LH fp'].mode().iloc[0] # scalar. if tie picks lower value
        seg_preds.loc[ind,"RH smoothM35"] = seg_preds.loc[s_ind:e_ind]['RH fp'].mode().iloc[0]
        
        s_ind = ind - 32 if ind>=33 else 1
        e_ind = ind + 32 if ind<=nFr-32 else nFr
        seg_preds.loc[ind,"LH smoothM65"] = seg_preds.loc[s_ind:e_ind]['LH fp'].mode().iloc[0] # scalar. if tie picks lower value
        seg_preds.loc[ind,"RH smoothM65"] = seg_preds.loc[s_ind:e_ind]['RH fp'].mode().iloc[0]
        if (ind+1%int(len(seg_preds.index)/5))==0:
            print(f"smoothed {ind+1}/{len(seg_preds.index)} frames")
    #
    # pickle seg_preds
    # with open(vidm+'last_seg_preds.txt', 'wb') as f:
    #     pickle.dump([vid, seg_preds,nFr],f)     
    
    seg_preds.to_csv(vidm +'_last_seg_preds.csv')
    # with open('last_seg_preds.txt', 'rb') as f:
    #     vid, seg_preds,nFr = pickle.load(f)
    
    # Create output video
    cap = cv.VideoCapture(vid)
    out_vid_fn = vid.split('.')[0] + '_' + time_str + '.avi' # base on orig vid + time made
    out_vid = cv.VideoWriter(out_vid_fn, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (fr_w, fr_h))
    frNum = 1
    while (cap.isOpened()) and frNum < 100000: # Read until video is completed again
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # model results # a pd.DataFrame with columns 'xmin','ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
            mr = all_bboxes[all_bboxes["Frame"]==frNum]
            fr_w = frame.shape[1]
            fr_h = frame.shape[0]
            # Apply all the bounding boxes using the network predictions
            for ind in range(len(mr.index)):
                # bbox = [xmin, ymin, xmax, ymax]
                bbox = [mr.iloc[ind,0], mr.iloc[ind,1], mr.iloc[ind,2], mr.iloc[ind,3]]
                bbox = [int(x) for x in bbox]
                #help(bbv.bbox_visualizer.draw_rectangle)
                #help(bbv.draw_rectangle)
                frame = bbv.draw_rectangle(frame, bbox)
                frame = bbv.add_label(frame, str(mr.iloc[ind,6])+': '+str(round(mr.iloc[ind,4],4)), bbox)
                
            # Attach ground truth usage label given frNum
            useLH = seg_preds.loc[frNum,"LHGT"]       
            useRH = seg_preds.loc[frNum,"RHGT"]
            
            # add in the usage info here
            useDict = {0:"T3",1:"T3",2:"T1",3:"T1",4:"T2",5:"T2",6:"T0",7:"T0"}
            sp_LH = useDict[seg_preds.loc[frNum,'LH smoothM65']] # smooth pred
            sp_RH = useDict[seg_preds.loc[frNum,'RH smoothM65']] # smooth pred
            title = f"GT: LH {str(useLH)} RH {str(useRH)} | SP: LH {sp_LH} RH {sp_RH}" 
            font, fontScale, fontColor, thick = (cv.FONT_HERSHEY_SIMPLEX,1,[255,0,0],4)
            cv.putText(frame, title, (int(fr_w / 50), int(fr_h / 15)), font, fontScale, fontColor, thick)
            
            # Save the resulting frame to the vid
            out_vid.write(frame)
            if (frNum%int(nFr/5))==0:
                print(f"wrote {frNum}/{nFr} frames to vid")
            # increment the frame number
            frNum += 1
        # Break the loop
        else:
            break
    
    # Plot temporal segmentation for the vid. One plot per smoothing approach
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(24,8))
    ax.set_yticks(np.arange(0,4,1), labels = ["LH GT","LH sp","RH GT","RH sp"])
    ax.set_ylim([-.75,3.75])
    ax.set_xlim([0,nFr])
    ax.set_xticks(np.arange(0,nFr,int(nFr/11)))
    ax.set_title(out_vid_fn.split('/')[-1])
    
    
    for mode_window in [["LH smoothM15","RH smoothM15"],["LH smoothM35","RH smoothM35"],["LH smoothM65","RH smoothM65"]]:
        for hand in mode_window:
            d = 1 if "LH" in hand else 3 
            av_color = ['r','r','g','g','y','y','b','b']
            colors = [av_color[int(p)] for p in seg_preds[hand]]
            for i in range(len(colors)): 
                ax.barh(d, height=.8, width=1, left = i, color=colors[i])
            
        for hand1 in ["LHGT","RHGT"]:
            d = 0 if "LH" in hand1 else 2 
            useColor = {"T0":'b',"T1":'g',"T2":'y',"T3":'r'}
            colors = [useColor[gt] for gt in seg_preds[hand1]]
            for i in range(len(colors)): 
                ax.barh(d, height=.8, width=1, left = i, color=colors[i])
        # Add legend for colors 
        blue_patch = mpatches.Patch(color='blue', label='T0')
        green_patch = mpatches.Patch(color='green', label='T1')
        yellow_patch = mpatches.Patch(color='yellow', label='T2')
        red_patch = mpatches.Patch(color='red', label='T3')
        ax.legend(handles=[blue_patch,green_patch,yellow_patch,red_patch])
        # Save the plots
        figName = vid.split('/')[-1].replace('.wmv','')[:5] + hand[-9:] + "usage_plot"
        plt.savefig(time_str+figName+".png")
    
    # When everything done, release the video capture object
    cap.release()
    out_vid.release()
    # Closes all the frames
    cv.destroyAllWindows()



