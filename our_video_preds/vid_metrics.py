# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:54:43 2022

@author: atver
"""
import numpy as np
import pandas as pd 
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score



videos = ["/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P022_balloon1.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P023_tissue2.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P024_balloon1.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P025_tissue2.wmv",
"/home/student/Desktop/hW1/Surgical-OD-HW1/videos/P026_tissue1.wmv"]

res = pd.DataFrame()

for vid in videos:
    vidm = vid.split('/')[-1][:4]
    seg_preds = pd.read_csv(vidm +'_last_seg_preds.csv')
    use2classLH = {"T0": 7, "T1": 3,"T2": 5, "T3": 1}
    use2classRH = {"T0": 6, "T1": 2,"T2": 4, "T3": 0}
    seg_preds["LHGT"] = [use2classLH[use] for use in seg_preds["LHGT"] ]
    seg_preds["RHGT"] = [use2classRH[use] for use in seg_preds["RHGT"] ]
    
    # drop any frame with no pred bbox for that hand !!
    seg_predsLHst = seg_preds[seg_preds["LH straight"] != '[]']
    seg_predsRHst = seg_preds[seg_preds["RH straight"] != '[]']
    # simplify/ homogenize the straight column same if GT if in the predicted array or else first pred
    seg_predsLHst["LH straight"] = [seg_predsLHst.loc[i,"LHGT"] \
                                if str(seg_predsLHst.loc[i,"LHGT"]) in seg_predsLHst.loc[i,"LH straight"] \
                                else int(seg_predsLHst.loc[i,"LH straight"][1]) for i in seg_predsLHst.index]
    seg_predsRHst["RH straight"] = [seg_predsRHst.loc[i,"RHGT"] \
                                if str(seg_predsRHst.loc[i,"RHGT"]) in seg_predsRHst.loc[i,"RH straight"] \
                                else int(seg_predsRHst.loc[i,"RH straight"][1]) for i in seg_predsRHst.index]
    
        
    print(vid)
    print("Straight")
    for Hand in ["L","R"]:
        print("Hand "+Hand)
        df_class = seg_predsLHst if Hand=="L" else seg_predsRHst
        y_true = list(df_class["LHGT"]) if Hand=="L" else list(df_class["RHGT"])
        y_pred = list(df_class["LH straight"]) if Hand=="L" else list(df_class["RH straight"])
        f1 = f1_score(y_true,y_pred,average='macro')
        print("Straight f1 macro: ", f1)
        res.loc[vid,"sf1_macro_"+Hand] = f1
        acc = accuracy_score(y_true, y_pred) # normalized
        print("Straight acc: ", acc)
        res.loc[vid,"sacc_"+Hand] = acc
        
    for cl in [0,1,2,3,4,5,6,7]:
        print("class: ",cl)
        Hand = "L" if cl%2 else "R"
        # straight
        df_class = seg_predsLHst[seg_predsLHst["LHGT"]==cl] if Hand=="L" else seg_predsRHst[seg_predsRHst["RHGT"]==cl]
        df_class = seg_predsLHst if Hand=="L" else seg_predsRHst # FP
        y_true = list(df_class["LHGT"]) if Hand=="L" else list(df_class["RHGT"])
        y_pred = list(df_class["LH straight"]) if Hand=="L" else list(df_class["RH straight"])
        prec = precision_score(y_true, y_pred,labels=[cl],average=None,zero_division=0)
        print('straight prec: ', prec)
        res.loc[vid,"sprec_"+str(cl)] = prec
        recall = recall_score(y_true, y_pred, average=None,labels=[cl], zero_division=0) # macro-noaccounting for label imbalance,micro-global, weighted=acc
        print('straight recall: ', recall)
        res.loc[vid,"srecall_"+str(cl)] = recall
        f1 = f1_score(y_true,y_pred, labels=[cl], average=None)
        print("Straight f1 class: ", f1)
        res.loc[vid,"sf1_"+str(cl)] = f1
     
    
    
    print("SmoothM65")
    for Hand in ["L","R"]:
        print("Hand "+Hand)
        df_class = seg_preds
        y_true = list(df_class["LHGT"]) if Hand=="L" else list(df_class["RHGT"])
        y_pred = list(df_class["LH smoothM65"]) if Hand=="L" else list(df_class["RH smoothM65"])
        f1 = f1_score(y_true,y_pred,average='macro')
        print("smoothM65 f1 macro: ", f1)
        res.loc[vid,"sm65_f1macro"+Hand] = f1
        acc = accuracy_score(y_true, y_pred) # normalized
        print("Smooth65 acc: ", acc)
        res.loc[vid,"sm65_acc"+Hand] = acc
        
    for cl in [0,1,2,3,4,5,6,7]:
        print("class: ",cl)
        Hand = "L" if cl%2 else "R"
        # smooth 65
        df_class = seg_preds#[seg_preds["LHGT"]==cl] if Hand=="L" else seg_preds[seg_preds["RHGT"]==cl]
        y_true = list(seg_preds[Hand + "HGT"])
        y_pred = list(seg_preds[Hand + "H smoothM65"])
        prec = precision_score(y_true, y_pred, labels=[cl],average=None, zero_division=0)
        print('Smooth65 prec: ', prec)
        res.loc[vid,"sm65prec_"+str(cl)] = prec
        recall = recall_score(y_true, y_pred, average=None,labels=[cl], zero_division=0) # macro-noaccounting for label imbalance,micro-global, weighted=acc
        print('Smooth65 recall: ', recall)
        res.loc[vid,"sm65recall_"+str(cl)] = recall
        f1 = f1_score(y_true,y_pred, labels=[cl], average=None)
        print("Smooth65 f1 class: ", f1)
        res.loc[vid,"sm65f1_"+str(cl)] = f1

    print("Fp")
    for Hand in ["L","R"]:
        print("Hand "+Hand)
        df_class = seg_preds
        y_true = list(df_class["LHGT"]) if Hand=="L" else list(df_class["RHGT"])
        y_pred = list(df_class["LH fp"]) if Hand=="L" else list(df_class["RH fp"])
        f1 = f1_score(y_true,y_pred,average='macro')
        print("fp f1 macro: ", f1)
        res.loc[vid,"FP_f1"+Hand] = f1
        acc = accuracy_score(y_true, y_pred) # normalized
        print("fp acc: ", acc)
        res.loc[vid,"FPacc_"+Hand] = acc
        
    for cl in [0,1,2,3,4,5,6,7]:
        print("class: ",cl)
        Hand = "L" if cl%2 else "R"
        # first pass
        df_class = seg_preds# [seg_preds["LHGT"]==cl] if Hand=="L" else seg_preds[seg_preds["RHGT"]==cl]
        y_true = list(seg_preds[Hand + "HGT"])
        y_pred = list(seg_preds[Hand + "H fp"])
        prec = precision_score(y_true, y_pred,labels=[cl],average=None,zero_division=0)
        print('fp prec binary: ', prec)
        res.loc[vid,"FPprec_"+str(cl)] = prec
        recall = recall_score(y_true, y_pred, average=None,labels=[cl],zero_division=0) # macro-noaccounting for label imbalance,micro-global, weighted=acc
        print('fp recall binary: ', recall)
        res.loc[vid,"FPrecall_"+str(cl)] = recall
        f1 = f1_score(y_true,y_pred, labels=[cl], average=None)
        print("fp f1 class: ", f1)
        res.loc[vid,"FPf1_"+str(cl)] = f1

res.to_csv("results.csv")


