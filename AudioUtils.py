#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
import subprocess as sp
import librosa
import progressbar
import random

# TODO : Some samples can't be recognizes by librosa (ex : cleanning step of split_samples())
#   - parse all samples with ffmpeg and save & replace each sample ???
# TODO : Print some activated layers for each label or export soungs with audible activated layers
# TODO : Print prediction curve for each laber for a specific song

# ======================
# AUDIO FILES MANAGEMENT
# ======================
        
def mp3_to_wav(path):
    
    """convert each mp3 file in path to wav"""
    
    # Path of FFMPEG
    if sys.platform == 'win32': FFMPEG_BIN = "C:\\FFmpegTool\\bin\\ffmpeg.exe"
    elif sys.platform == 'linux': FFMPEG_BIN = 'ffmpeg'
    
    for folder in os.listdir(path):
        print('Process : ' + folder)
        for file in os.listdir(os.path.join(path, folder)):
            sp.call([FFMPEG_BIN, '-y', '-i', os.path.join(path, folder, file),
                     '-ar','22050',
                     os.path.join(path, folder, file.split('.')[0] + '.wav')],shell=False)
            os.remove(os.path.join(path, folder, file))
        print('     ' + folder + '... Done')
            
def delete_long_short_samples(path, minlength = 1.5, maxlength = 3):
    
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path,folder)):
            sample = os.path.join(path,folder,file)
            if librosa.get_duration(filename=sample) < minlength:
                os.remove(sample)
            elif librosa.get_duration(filename=sample) > maxlength:
                os.remove(sample)
        print('Instrument : ' + folder + ' ... DONE!')
        
def norm_silenceSplit(path, minlength = 2.5):
    
    """ Normalise loudness
        Split by silences
    """
    
    if sys.platform == 'win32': SOX_BIN = "C:\\sox-14-4-2\\sox.exe"
    elif sys.platform == 'linux': SOX_BIN = 'sox'
    
    for folder in os.listdir(path):
        start_size = len(os.listdir(os.path.join(path, folder)))
        for file in os.listdir(os.path.join(path, folder)):
            
            fmt = '.' + file.split('.')[-1]
            
            inp = os.path.join(path,folder,file)
            tmp = os.path.join(path, folder, file.split(fmt,1)[0] + '_normed' + fmt)
            out = os.path.join(path, folder, file.split(fmt,1)[0] + '_' + fmt)

            # Normalize volume
            trim = [SOX_BIN,inp,tmp,'norm']
            ctrl = sp.call(trim,shell=False)
            os.remove(inp)
            if ctrl > 0: print('Error on : ' + file)
            
            #if folder == 'Bass':
            #    db = '-40d'
            #else: db = '2%'
            db = '-60d'
            
            #   Split by silences ...
            trim = [SOX_BIN,tmp,out,
                    'silence','1','2.5',str(db),'1','0.5',str(db), ':','newfile',':','restart']
            ctrl = sp.call(trim,shell=False)
            os.remove(tmp)
            if ctrl > 0: print('Error on : ' + file)
        
        end_size = len(os.listdir(os.path.join(path, folder)))
        print('Instrument : ' + folder + ' ... DONE!', '\n     Nb samples ini : ',start_size,
              '\n     Nb samples fin : ',end_size)
        
def split_samples(path, cliplength = 2.5):
    
    if sys.platform == 'win32': FFMPEG_BIN = "C:\\FFmpegTool\\bin\\ffmpeg.exe"
    elif sys.platform == 'linux': FFMPEG_BIN = 'ffmpeg'
    
    for folder in os.listdir(path):
        start_size = len(os.listdir(os.path.join(path, folder)))
        if start_size > 0:
            for file in os.listdir(os.path.join(path, folder)):
                
                fmt = '.' + file.split('.')[-1]
                
                inp = os.path.join(path, folder, file)
                out = os.path.join(path, folder, file.split(fmt,1)[0] + '_%03d' + fmt)
                
                if librosa.get_duration(filename=inp) >= cliplength:
                
                    trim = [FFMPEG_BIN,'-i',inp,
                            '-f','segment',
                            '-segment_time',str(cliplength),
                            '-c','copy',
                            out]
                    sp.call(trim,shell=False)
                    os.remove(inp)
            
            end_size = len(os.listdir(os.path.join(path, folder)))
            print('Instrument : ' + folder + ' ... DONE!', '\n     Nb samples ini : ',start_size,
                  '\n     Nb samples fin : ',end_size)

def merge_audio_bases(sourcepath, targetpath):
    
    for folder in os.listdir(sourcepath):
        for file in os.listdir(os.path.join(sourcepath, folder)):
            
            if not os.path.exists(os.path.join(targetpath, folder)):
                os.makedirs(os.path.join(targetpath, folder))
            
            inp = os.path.join(sourcepath, folder, file)
            out = os.path.join(targetpath, folder, file)
            
            if not os.path.exists(out):
                os.rename(inp,out)
        print('Instrument : ' + folder + ' ... DONE!')

# =================
# DATA AUGMENTATION
# =================

def samples_augmentation(path, factor = 5):
    
    SOX_BIN = "C:\\sox-14-4-2\\sox.exe"
    
    def list_flatten(A):
        flat = []
        for i in A:
            if isinstance(i,list): flat.extend(list_flatten(i))
            else: flat.append(i)
        return flat
    
    for folder in os.listdir(path):
        start_size = len(os.listdir(path + '\\' + folder))
        for file in os.listdir(path + '\\' + folder):
            for i in range(factor):
                
                inp = path + '\\' + folder + '\\' + file
                out = path + '\\' + folder + '\\' + file.split('.wav',1)[0] + '_' + str(i) + '.wav'
                
                effect = ['pitch',str(random.randint(-200,200)),'dither',
                           'tempo',str(random.uniform(0.70,1.5)),'dither']
                if random.choice(range(10)) > 6: 
                    effect.append(['overdrive',str(random.randint(0,15))])
                if random.choice(range(10)) > 4: 
                    effect.append(['reverb',str(random.randint(0,60)),str(random.randint(0,60)),str(random.randint(0,60)),str(random.randint(0,0))])
                effect.append(['norm'])
                effect = list_flatten(effect)
                command = [SOX_BIN,inp,out,effect]
                flat_command = list_flatten(command)
                sp.call(flat_command,shell=False)
        end_size = len(os.listdir(path + '\\' + folder))
        print('Instrument : ' + folder + ' ... DONE!', '\n     Nb samples ini : ',start_size,
              '\n     Nb samples fin : ',end_size)

# ===================
# FEATURES GENERATION
# ===================
        
#############################################################
#       Extraction des Features Audio par STFT - MFCC       #
#              D'après HAN, KIM ET LEE (2016)               #
#############################################################

def compute_mfcc(file_name):
 
    """ Compute a mel-spectrogram by file """

    # mel-spectrogram parameters
    SR = 12000
    OFFSET = 1
    DURA = 3
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256

    src, sr = librosa.load(file_name,sr=SR,offset=OFFSET,duration=DURA,mono=True)  # whole signal
    
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short, add zeros
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long, cut begining and end equally
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)

    return ret

def compute_chroma(file_name):
    
    SR = 12000
    OFFSET = 0.25
    DURA = 1.5

    src, sr = librosa.load(file_name,sr=SR,offset=OFFSET,duration=DURA,mono=True) 
    
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    
    if n_sample < n_sample_fit:  # if too short, add zeros
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long, cut begining and end equally
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    
    ret = librosa.feature.chroma_stft(y=src, sr=sr, n_chroma=12, 
                                      n_fft=8192)
    ret = np.mean(ret,axis=1)
    return ret

def compute_multi_mfcc(file_name,limit):
    
    """ Compute multiple mel-spectrograms by file and return start time for each one """

    # mel-spectrogram parameters
    SR = 12000
    OFFSET = 0.25
    DURA = 1.5
    N_FFT = 1024
    N_MELS = 96
    HOP_LEN = 256
    
    # MFCC overlaping
    OVERLAP = 0.5
    
    # Simplify librosa's functions name
    logam = librosa.logamplitude # Convert powered-scaled spect (square of amplitude) to decibel-scaled
    melgram = librosa.feature.melspectrogram # Compute mel-scaled spect (frequency-oriented scaling)
    
    # Import .wav
    src, sr = librosa.load(file_name,sr=SR,offset=OFFSET,mono=True,duration = 30)
    
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short, complete with zeros
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
        melfilt = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                n_fft=N_FFT, n_mels=N_MELS)**2,ref_power=1.0)
        ret = melfilt[np.newaxis, :]
    elif n_sample >= n_sample_fit:  # if too long, compute as much MFCC as possible
        i = n_sample_fit
        j=0
        while (i <= n_sample) and (j < limit):
            signal = src[i-n_sample_fit:i,]
            if j == 0:
                melfilt = logam(melgram(y=signal, sr=SR, hop_length=HOP_LEN,
                                        n_fft=N_FFT, n_mels=N_MELS)**2,ref_power=1.0)
                ret = melfilt[np.newaxis, :]
            else: 
                melfilt = logam(melgram(y=signal, sr=SR, hop_length=HOP_LEN,
                                        n_fft=N_FFT, n_mels=N_MELS)**2,ref_power=1.0)
                melfilt = melfilt[np.newaxis, :]
                ret = np.concatenate((ret,melfilt),axis = 0)
            i = i + int((1-OVERLAP)*n_sample_fit)
            j = j + 1

    return ret

def compute_multi_chroma(file_name,limit):
    
    """ Compute multiple mel-spectrograms by file and return start time for each one """

    # chromagram parameters
    SR = 12000
    OFFSET = 0.25
    DURA = 0.5
    
    # MFCC overlaping
    OVERLAP = 0
    
    # Import .wav
    src, sr = librosa.load(file_name,sr=SR,offset=OFFSET,mono=True,duration = 30)
    
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short, complete with zeros
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
        chroma = librosa.feature.chroma_stft(y=src, sr=sr, n_chroma=12, 
                                          n_fft=16384)
        chroma = np.mean(chroma,axis=1)
        ret = chroma[np.newaxis, :]
    elif n_sample >= n_sample_fit:  # if too long, compute as much MFCC as possible
        i = n_sample_fit
        j=0
        while (i <= n_sample) and (j < limit):
            signal = src[i-n_sample_fit:i,]
            if j == 0:
                chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12, 
                                                     n_fft=8192)
                chroma = np.mean(chroma,axis=1)
                ret = chroma[np.newaxis, :]
            else: 
                chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12, 
                                                     n_fft=8192)
                chroma = np.mean(chroma,axis=1)
                chroma = chroma[np.newaxis, :]
                ret = np.concatenate((ret,chroma),axis = 0)
            i = i + int((1-OVERLAP)*n_sample_fit)
            j = j + 1


    return ret

def compute_mfcc_from_path(path,save_path,effect):
    
    """ Extract MFCC + STFT for each sample in path and save output in save_path """
    
    filecounter = 0
    nb_files = len(os.listdir(path + '\\'))
    with progressbar.ProgressBar(max_value=nb_files) as bar:
        for file in os.listdir(path + '\\'):
            
            mfcc = compute_mfcc(path + file,effect)
            
            if filecounter == 0:
                nb_mfcc = mfcc.shape[0]
            else:
                nb_mfcc = np.vstack((nb_mfcc,mfcc.shape[0]))
            
            np.save(save_path + str(file[0:(len(file)-4)]),mfcc)

            filecounter += 1
            bar.update(filecounter)
            
    return nb_mfcc

def compute_multi_mfcc_from_path(path,save_path,max_mfcc):
    
    """ Extract multiples MFCC + STFT for each sample in path and save output in save_path """
    
    filecounter = 0
    nb_files = len(os.listdir(path + '\\'))
    with progressbar.ProgressBar(max_value=nb_files) as bar:
        for file in os.listdir(path + '\\'):
            
            mfcc = compute_multi_mfcc(path + file, limit = max_mfcc)
            
            if filecounter == 0:
                nb_mfcc = mfcc.shape[0]
            else:
                nb_mfcc = np.vstack((nb_mfcc,mfcc.shape[0]))
            
            np.save(save_path + str(file[0:(len(file)-4)]),mfcc)

            filecounter += 1
            bar.update(filecounter)
    np.save(save_path + 'mfcc_by_file',nb_mfcc)
    
def compute_mfcc_from_samples(path,save_path,n):
    
    """ Extract MFCC + STFT for each sample in path and save output in save_path """
    
    labels = np.empty([0,3], dtype='str')
    selected_samples = np.empty([0,2], dtype='str')
    j = 0
    for folder in os.listdir(path):
        if len(os.listdir(path + '\\' + folder)) > 1000:
            for i in range(0,n):
                file = random.choice(os.listdir(path + '\\' + folder))
                mfcc = compute_mfcc(path + '/' + folder + '\\' + file)
                    
                labels = np.vstack((labels,np.array([str(j).zfill(5),folder,1])))
                selected_samples = np.vstack((selected_samples,np.array([str(j).zfill(5),file])))
                np.save(save_path + '\\' + str(j).zfill(5),mfcc)

                j += 1

        print('Instrument : ' + folder + ' ... DONE!')
    
    np.savetxt('randomSlabels_files.csv',selected_samples,delimiter=",",fmt="%s")
    
    import pandas as pd
    df = pd.DataFrame(labels,columns = ['id','variable','value'])
    df = df.pivot(index = 'id',columns = 'variable',values='value')
    df = df.fillna(0) 
    df.to_csv('randomSlabels' + '.csv',sep=',')
