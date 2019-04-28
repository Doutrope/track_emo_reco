# -*- coding: utf-8 -*-

import os

# FORCE CPU 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import progressbar
import pandas as pd
import numpy as np
import json 
import sys 
import subprocess as sp
import AudioUtils

#### Smoothing function
def smoothize(x,p):
    out=np.empty(len(x))
    for i in range(len(x)):
        if i-p < 0: start = 0
        else: start = i-p
        out[i] = str(round(np.mean(x[(start):(i+p)]),2))            
    return out
#### Ffmpeg path
if sys.platform == 'win32': FFMPEG_BIN = "C:\\FFmpegTool\\bin\\ffmpeg.exe"
elif sys.platform == 'linux': FFMPEG_BIN = 'ffmpeg'

# ===========
# PREPARATION
# ===========

# GET URLS AND IDS
metadata = pd.read_csv(os.path.join('data','deezer_metadata','metadata.csv'))
metadata['id'][0]
scrap_source = {}
for i in range(len(metadata)):
#for i in range(10):
    ids = metadata['id'][i]
    prev = metadata['preview'][i]
    if prev != 'unknown':
        scrap_source[ids] = prev
del metadata,i,ids,prev

# LOAD MODELS
arousal = keras.models.load_model(os.path.join('models','model_arou_testRMSE.h5'))
valence = keras.models.load_model(os.path.join('models','model_valen_testRMSE.h5'))

# LOAD ALREADY COMPUTED PREDS
arousal_predictions = np.load(os.path.join('data','tmp','arousal_predictions.npy'))
valence_predictions = np.load(os.path.join('data','tmp','valence_predictions.npy'))

already_computed = arousal_predictions[:,0].tolist()
tracks_ids = list(scrap_source.keys())
for key in tracks_ids:
    if str(key) in already_computed: del scrap_source[key]
del already_computed,key

# ===========================
# DOWNLOAD - PREDICT - DELETE
# ===========================

#import time
import urllib

pb=0
computed = 0
with progressbar.ProgressBar(max_value=len(scrap_source)) as bar:
    for i in scrap_source.keys():
        try:
            
            # DOWNLOAD SONG
            r = urllib.request.urlopen(scrap_source[i])
            data = r.read()
            with open(os.path.join('data','tmp',str(i) + '.mp3'),'wb') as f:
                f.write(data)
            
            # CONVERT TO WAV
            sp.call([FFMPEG_BIN, '-y', '-i', os.path.join('data','tmp',str(i) + '.mp3'),
                     '-ar','22050',
                     os.path.join('data','tmp',str(i) + '.wav')],shell=False)
            
            # MODELIZE EMOTION
            tmp_mfcc = AudioUtils.compute_multi_mfcc(os.path.join('data','tmp',str(i) + '.wav'),limit = 35)
            tmp_chroma = AudioUtils.compute_multi_chroma(os.path.join('data','tmp',str(i) + '.wav'),limit = 60)
            
            for j in range(tmp_mfcc.shape[0]):
                tmp_mfcc[j] = (tmp_mfcc[j] - np.mean(tmp_mfcc[j]))/(np.std(tmp_mfcc[j])+1e-7) 
        
            tmp_mfcc = tmp_mfcc.reshape(tmp_mfcc.shape[0], tmp_mfcc.shape[1], tmp_mfcc.shape[2], 1)
            tmp_chroma = tmp_chroma.reshape(1,tmp_chroma.shape[0],tmp_chroma.shape[1])
            
            pred_arou = arousal.predict(tmp_mfcc).T
            pred_valen = valence.predict(tmp_chroma)
            
            pred_valen = np.reshape(pred_valen,(1,60))
            
            pred_arou[0] = smoothize(pred_arou[0],3)
            pred_valen[0] = smoothize(pred_valen[0],3)
            
            pred_arou = np.concatenate((np.array([[str(i)]]),
                                        pred_arou),axis=1)
            pred_valen = np.concatenate((np.array([[str(i)]]),
                                         pred_valen),axis=1)
        
        
            if pb == 0:
                arousal_predictions = pred_arou
                valence_predictions = pred_valen
            else:
                arousal_predictions = np.concatenate((arousal_predictions,pred_arou),axis = 0)
                valence_predictions = np.concatenate((valence_predictions,pred_valen),axis = 0)
                
            if computed % 1000 == 0:
                np.save(os.path.join('data','tmp','arousal_predictions.npy'),arousal_predictions)
                np.save(os.path.join('data','tmp','valence_predictions.npy'),valence_predictions)
            
            # REMOVE SONG
            os.remove(os.path.join('data','tmp',str(i) + '.mp3'))
            os.remove(os.path.join('data','tmp',str(i) + '.wav'))
        except:
            pass
        
        pb+=1
        computed += 1
        bar.update(pb)

np.save(os.path.join('data','tmp','arousal_predictions.npy'),arousal_predictions)
np.save(os.path.join('data','tmp','valence_predictions.npy'),valence_predictions)
		
#arou_pred = arousal_predictions[:,1:]
#arou_pred = pd.DataFrame(arousal_predictions[:,1:],
#                         index=arousal_predictions[:,0],
#                         dtype = 'float32').T
#valen_pred = valence_predictions[:,1:]
#valen_pred = pd.DataFrame(valence_predictions[:,1:],
#                         index=valence_predictions[:,0],
#                         dtype = 'float32').T
#
#mini = min(arou_pred.min().tolist())
#maxi = max(arou_pred.max().tolist())
#arous_preds = {}
#for column in arou_pred:
#    arous_preds[column] = [round(100*(x-mini)/(maxi-mini),0) for x in arou_pred[column].tolist()]
#    
#mini = min(valen_pred.min().tolist())
#maxi = max(valen_pred.max().tolist())
#valen_preds = {}
#for column in valen_pred:
#    valen_preds[column] = [round(100*(x-mini)/(maxi-mini),0) for x in valen_pred[column].tolist()]
#    
## Merge dictionnaries
#predictions = {}
#for track in arous_preds:
#    predictions[track] = {'arousal': arous_preds[track],
#                          'valence': valen_preds[track]}
#    
## ===============================
## TRANSFORM DATA FOR RADIAL GRAPH
## ===============================    
#import math
#    #import
#with open(os.path.join('data','emo_predictions.json'), encoding='utf-8') as fh:
#    predictions = json.load(fh)
#    
#    #interpolate for arrays having same size
#from scipy.interpolate import interp1d
#'''
#y = predictions['1Kilo-Deixe_Me_Ir_(Acústico).mp3']['arousal']
#x = np.linspace(0, 60, num=35, endpoint=True)
#intrerp = interp1d(x, y, kind='quadratic')
#xnew = np.linspace(0, 60, num=60, endpoint=True)
#newval = intrerp(xnew)
#'''
#
#emotion_bins = {0: 'Deactivated',
#        1: 'Fatigued',2: 'Bored',3: 'Depressed',4: 'Sad',
#        5: 'Unpleasant',6: 'Upset',7: 'Stressed',8: 'Nervous',
#        9: 'Tense',10: 'Activated',11: 'Alert',12: 'Excited',
#        13: 'Elated',14: 'Happy',15: 'Pleasant',16: 'Contented',
#        17: 'Serene',18: 'Relaxed',19: 'Calm',20: 'Deactivated'}
#emo_ordered = ['Activated','Tense','Nervous','Stressed','Upset','Unpleasant',
#               'Sad','Depressed','Bored','Fatigued','Deactivated','Calm',
#               'Relaxed','Serene','Contented','Pleasant','Happy','Elated',
#               'Excited','Alert']
#
#radial_preds = {}
#for i in predictions:
#    
#    x = predictions[i]['valence']
#    y = predictions[i]['arousal']
#    
#    # Interpolate Arousal to get same array sizes
#    tmp = np.linspace(0, 60, num=35, endpoint=True)
#    intrerp = interp1d(tmp, y, kind='quadratic')
#    xnew = np.linspace(0, 60, num=60, endpoint=True)
#    newval = intrerp(xnew)
#    
#    y = newval.tolist()
#    
#    # Normalize emotions values
#    x = [(val - 50)/50 for val in x]
#    y = [(val - 50)/50 for val in y]
#
#    # convert arousal/valence to radial emotion/intensity
#    angles = []
#    intens = []
#    for j in range(60):
#        
#        intens_cart = math.sqrt(x[j]**2+y[j]**2)
#        angle = math.atan2(x[j],y[j])
#        
#        if abs(angle) <= math.pi/2:
#            angle_red = abs(angle)
#        else: angle_red = abs(angle) - math.pi/2
#        
#        if angle_red <= math.pi/4:
#            intens_max = 1/math.cos(angle_red)
#        else: intens_max = 1/math.cos(math.pi/2-angle_red)
#        
#        inten = intens_cart/intens_max
#        
#        angles.append(angle)
#        intens.append(inten)
#
#    # discretize angular value to feet emotions    
#    bins = np.arange(-math.pi - 2*math.pi/40,
#                     math.pi + 3*math.pi/40,
#                     2*math.pi/20)
#    digitized = np.digitize(angles, bins)
#    emo = [emotion_bins[x-1] for x in digitized]
#    
#    emo = np.array(emo)
#    intens = np.array(intens)
#
#    d = []
#    for emotion in emo_ordered:
#        if emotion in emo:
#            d.append({'axis':emotion,'value':round(np.mean(intens[emo == emotion]),2)})
#        else: d.append({'axis':emotion,'value': 0})
#        
#    radial_preds[i] = d
#    
#with open('data/' + 'emo_radial_predictions.json', 'w',encoding='utf8') as fp:
#    json.dump(radial_preds, fp, ensure_ascii=False)