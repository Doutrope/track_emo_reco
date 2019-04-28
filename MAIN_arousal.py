


            ##########################################
            #       Modelize Valence & Arousal       #
            #             with ConvNets              #
            ##########################################


# REFERENCES : 
#   - Aljanaki, Yang & Soleymani : Developing a benchmark for emotional analysis of music (2016)
#   - Malik, Adavanne, Drossos, Virtanen, Ticha, Jarina : 
#       Stacked convolutional and recurrent neural networks for music emotion recognition (2017)
# 
# Best scores with convnets in litterature : 
#   - Arousal : RMSE = 0.202 +/- 0.007
#   - Valence : RMSE = 0.267 +/- 0.003

# Possible ways to increase learning database :
#   - Randomly modulate pitch
#   - Randomly modulate volume

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import progressbar

# Set working directory and import functions in AudioUtils.py
os.chdir('C:\\Users\\sacha\\Documents\\Projets\\DeepLearning\\\keras-EmoReco\\')
import AudioUtils

# ============================
# SET SEED FOR REPRODUCIBILITY
# ============================

np.random.seed(123)

# ====================
# IMPORT AND PROCESS Y
# ====================

import pandas as pd
valence = pd.read_csv(os.path.join('data','annotations','valence.csv'))
valence = valence.values
arousal = pd.read_csv(os.path.join('data','annotations','arousal.csv'))
arousal = arousal.values

# select seconds 15 to 45 and emotion to modelize
emo = np.take(arousal,np.array(list(range(60))) + 1,1)

# for each sample, take the average value for each window of 1.5 second, with overlap of 2/3
i=0
while i + 3 <= emo.shape[1]-1:
    tmp = np.average(emo[:,np.array(list(range(4))) + i],
                          axis=1)
    tmp = np.column_stack(tmp)
    if i == 0: 
        y = tmp
    else:
        y = np.concatenate((y,tmp),axis=0)
    i += 1
y = np.transpose(y)
y = y[:,:-1]

del [arousal,valence,tmp] 

# =============
# IMPORT MFCC's
# =============

# input MFCC dimensions
nbmfcc, img_rows, img_cols = np.load('data\\DEAM_mfccs\\' + random.choice(os.listdir('data\\DEAM_mfccs\\'))).shape

# Import MFCCs
printcounter = 0
gpecounter = 1
nb_files = len(os.listdir('data\\DEAM_mfccs\\'))
i=0
x = np.zeros((nb_files,nbmfcc,img_rows,img_cols))
with progressbar.ProgressBar(max_value=len(os.listdir('data\\DEAM_mfccs\\'))) as bar:
    for file in os.listdir('data\\DEAM_mfccs\\'):
        x[i,:,:,:] = np.load('data\\DEAM_mfccs\\' + file)
        i+=1
        bar.update(i)
        
# ======================
# TRAIN - TEST - RESHAPE
# ======================

# Sample train test
indices = np.random.permutation(x.shape[0])
training_idx, test_idx = indices[:int(x.shape[0]*0.7)], indices[int(x.shape[0]*0.7):]
x_train, x_test = x[training_idx,:], x[test_idx,:]
y_train, y_test = y[training_idx], y[test_idx]

del [x,y]

# Reshape arrays
x_train = np.reshape(x_train,(x_train.shape[0]*x_train.shape[1],
                              x_train.shape[2],
                              x_train.shape[3]))
x_test = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],
                            x_test.shape[2],
                            x_test.shape[3]))
y_train = np.reshape(y_train,(y_train.shape[0]*y_train.shape[1]))
y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]))

# Once train and test separated by song, reshuffle x and y and sample
indices_train = np.random.permutation(x_train.shape[0])
sample_train_idx = indices_train
x_train, y_train = x_train[sample_train_idx,:], y_train[sample_train_idx]

indices_test = np.random.permutation(x_test.shape[0])
sample_test_idx = indices_test
x_test, y_test = x_test[sample_test_idx,:], y_test[sample_test_idx]

# ====================
# KERAS INITIALISATION
# ====================

# ConvNets accepts 2 data formats according to Keras backend :
#   - If channel last, backend takes data as [rows, cols, channels]
#   - If channel first, backend takes data as [channels, rows, cols]

if K.image_data_format() == 'channels_first': 
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

batch_size = 128
num_classes = 1
epochs = 100

# ==============
# NORMALISATIONS
# ==============
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#x_train_min, x_train_max = np.amin(x_train),np.amax(x_train-np.amin(x_train))
#x_test_min, x_test_max = np.amin(x_test),np.amax(x_test-np.amin(x_test))

#x_train = (x_train - x_train_min)/x_train_max
#x_test = (x_test - x_test_min)/x_test_max

for i in range(x_train.shape[0]):
    x_train[i] = (x_train[i] - np.mean(x_train[i]))/(np.std(x_train[i])+1e-7) 
for i in range(x_test.shape[0]):
    x_test[i] = (x_test[i] - np.mean(x_test[i]))/(np.std(x_test[i])+1e-7)

#y_train -= np.amin(y_train)
#y_train /= np.amax(y_train)
#y_test -= np.amin(y_test)
#y_test /= np.amax(y_test)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ======================
# MODEL 1 (RMSE = 0.189)
# ======================

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='linear'))

model.summary()

# ==================================
# MODEL 2 : SIMPLIER (RMSE = 0.1878)
# ==================================

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='linear'))

model.summary()

# ====================================
# MODEL 3 : SIMPLIERER (RMSE = 0.1876)
# ====================================

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (10, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (10, 10), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='linear'))

model.summary()

# =====================================
# MODEL 3 : TIME DISTRIBUTED - TO DO...
# =====================================

#model = Sequential()
#model.add(Conv2D(8, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(num_classes, activation='linear'))
#
#model.summary()

# ========
# LEARNING
# ========

ES = [EarlyStopping(monitor='val_loss',patience=3,verbose=2,mode='auto')]

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.adam(),
              metrics=['mean_squared_error'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=ES)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test MSE:', score[1])


# ======================
# PLOTS LEARNING HISTORY
# ======================

#   L'accuracy sur l'échantillon test est parfois supérieure au train, du fait des mécanismes de régularisation, actifs
#   uniquement lors des phases d'apprentissage (la couche dropout est par exemple inactive lors de la prédiction sur train)
#   pareil pour le coût d'apprentissage, moyenne des coûts d'une itération, supérieur au coût de test, calculé à la fin d'une itération). 
#   Train et test accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Train et test loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ===========
# PREDICTIONS
# ===========

prediction = model.predict(x_test)

# ==========
# SAVE MODEL
# ==========

# Save model and weightings
model.save('C:\\Users\\sacha\\Documents\\Projets\\DeepLearning\\keras-EmoReco\\models\\model_arou_testRMSE.h5')

# Save model architecture without weightings in json format
with open('C:\\Users\\sacha\\Documents\\Projets\\DeepLearning\\keras-EmoReco\\models\\model_arou_testRMSE_architecture.json', 'w') as f:  
    f.write(model.to_json())  
    
# Load model and weightings
model = keras.models.load_model('models/model_arou_testRMSE.h5')

# ===========================
# TEST MODEL ON REGULAR SONGS
# ===========================

#### Smoothing function
def smoothize(x,p):
    out=np.empty(len(x))
    for i in range(len(x)):
        if i-p < 0: start = 0
        else: start = i-p
        out[i] = str(round(np.mean(x[(start):(i+p)]),2))            
    return out

i=0
with progressbar.ProgressBar(max_value=len(os.listdir(os.path.join('data','deezer_wav')))) as bar:
    for file in os.listdir(os.path.join('data','deezer_wav')):
        
        tmp_mfcc = AudioUtils.compute_multi_mfcc(os.path.join('data','deezer_wav',file),limit = np.inf)
        tmp_chroma = AudioUtils.compute_multi_chroma(os.path.join('data','deezer_wav',file),limit = np.inf)
        # Norm by train tracks
        #test_mfcc = (test_mfcc - x_train_min)/x_train_max
        # Norm by track itself
        #test_mfcc = (test_mfcc - np.mean(test_mfcc))/np.std(test_mfcc)
        # Norm each mfcc
        for j in range(tmp_mfcc.shape[0]):
            tmp_mfcc[j] = (tmp_mfcc[j] - np.mean(tmp_mfcc[j]))/(np.std(tmp_mfcc[j])+1e-7) 
        
        
        tmp_mfcc = tmp_mfcc.reshape(tmp_mfcc.shape[0], img_rows, img_cols, 1)
        tmp_chroma = tmp_chroma.reshape(1,tmp_chroma.shape[0],tmp_chroma.shape[1])
        
        pred_arou = model.predict(tmp_mfcc).T
        pred_valen = model.predict(tmp_chroma)
        
        pred_arou[0] = smoothize(pred_arou[0],3)
        
        pred_arou = np.concatenate((np.array([[file]]),
                                    pred_arou),axis=1)
        
        if i == 0:
            songs_predictions = pred_arou
        else:
            songs_predictions = np.concatenate((songs_predictions,pred_arou),axis = 0)
        
        i += 1
        bar.update(i)
        
test_pred = songs_predictions[:,1:]
test_pred = pd.DataFrame(songs_predictions[:,1:],
                         index=songs_predictions[:,0],
                         dtype = 'float32')
#test_pred.columns = np.arange(15.75, 23, 0.5).astype(str)

test_pred.to_csv(os.path.join('data','DeezerSongs_arousal.csv'),sep=',')
test_pred.to_csv(os.path.join('data','DeezerSongs_arousal.csv'),sep=',')
test_pred = pd.read_csv(os.path.join('data','DeezerSongs_arousal.csv'),index_col = 0)

#select caracteristic sample by mean arousal and std
tmp = test_pred
tmp['mean'] = tmp.mean(axis=1)
tmp['std'] = tmp.std(axis=1)

top_arr = tmp.sort_values('mean',ascending = False).head(15).index.values.tolist()
down_arr = tmp.sort_values('mean',ascending = True).head(15).index.values.tolist()
var_arr = tmp.sort_values('std',ascending = False).head(15).index.values.tolist()

selected = var_arr + down_arr + top_arr
selected = list(set(selected))

# =======================
# PLOT PREDICTIONS CURVES
# =======================

# D3 arrange

test_pred = test_pred.drop(['mean','std'],axis=1)
time_ser = test_pred.T[selected]
time_ser.to_csv(os.path.join('data','DeezerSongs_arousal_ts.csv'),sep=',')

time_ser = pd.read_csv(os.path.join('data','DeezerSongs_arousal_ts.csv'),index_col=0)

#time_ser['seconds'] = time_ser.index

mini = min(time_ser.min().tolist())
maxi = max(time_ser.max().tolist())

arou_preds = {}
for column in time_ser:
    arou_preds[column] = [round(100*(x-mini)/(maxi-mini),0) for x in time_ser[column].tolist()]

import json
with open('data/' + 'predictions.json', 'w') as fp:
    json.dump(arou_preds, fp)

# COMPUTE JS CMDS AS SHLAG
 
cmds = ['<option value="'+x+'">'+x.split('.')[0]+'</option>' for x in arou_preds.keys()]

with open('js.txt', mode='wt', encoding='utf-8') as f:
    f.write('\n'.join(cmds))

    
    


#### PLOT EXAMPLE ####
# inspired by https://matplotlib.org/gallery/showcase/bachelors_degrees_by_gender.html

gender_degree_data = time_ser


# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# You typically want your plot to be ~1.33x wider than tall. This plot
# is a rare exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 14))

# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
ax.set_xlim(0, 14)
ax.set_ylim(-0.65, 0.65)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
#plt.xticks(np.arange(14, 23, 1), fontsize=14)
#plt.yticks(np.arange(-0.60, 0.70, 0.1), fontsize=14)
#ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
#ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted.
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='on', left='off', right='off', labelleft='on')

# Now that the plot is prepared, it's time to actually plot the data!
# Note that I plotted the majors in order of the highest % in the final year.
majors = ['pop71.wav','pop43.wav','pop21.wav','rock37.wav','electronic56.wav',
          'electronic40.wav','electronic8.wav','classical2.wav','pop56.wav','electronic18.wav',
          'pop93.wav','rock65.wav','electronic90.wav','pop10.wav','pop14.wav',
          'pop16.wav','pop18.wav','electronic45.wav','rock19.wav','electronic15.wav']

y_offsets = {'pop71.wav': 0.49,
'pop43.wav': 0.47,
'pop21.wav': 0.5,
'rock37.wav': 0.51,
'electronic56.wav': 0.48,
'electronic40.wav': 0.5,
'electronic8.wav': 0.5,
'classical2.wav': 0.5,
'pop56.wav': 0.5,
'electronic18.wav': 0.5,
'pop93.wav': 0.5,
'rock65.wav': 0.51,
'electronic90.wav': 0.49,
'pop10.wav': 0.49,
'pop14.wav': 0.49,
'pop16.wav': 0.49,
'pop18.wav': 0.49,
'electronic45.wav': 0.48,
'rock19.wav': 0.51,
'electronic15.wav': 0.49,}

for rank, column in enumerate(majors):
    
    # Plot each line separately with its own color.

    line = plt.plot(gender_degree_data['seconds'],
                    gender_degree_data[column],
                    lw=2.5,
                    color=color_sequence[rank])

    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    y_pos = np.array(gender_degree_data[column])[-1] - 0.5

    if column in y_offsets:
        y_pos += y_offsets[column]

    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    plt.text(14.25, y_pos, column, fontsize=14, color=color_sequence[rank])

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
fig.suptitle('Arousal on caracteristic tracks', fontsize=15, ha='center')

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')
plt.show()













#### PLOT EXAMPLE ####


gender_degree_data = pd.read_csv('percent_bachelors_degrees_women_usa.csv')


# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# You typically want your plot to be ~1.33x wider than tall. This plot
# is a rare exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 14))

# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
ax.set_xlim(1969.5, 2011.1)
ax.set_ylim(-0.25, 90)

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(range(1970, 2011, 10), fontsize=14)
plt.yticks(range(0, 91, 10), fontsize=14)
ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted.
plt.tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='on', left='off', right='off', labelleft='on')

# Now that the plot is prepared, it's time to actually plot the data!
# Note that I plotted the majors in order of the highest % in the final year.
majors = ['Health Professions', 'Public Administration', 'Education',
          'Psychology', 'Foreign Languages', 'English',
          'Communications and Journalism', 'Art and Performance', 'Biology',
          'Agriculture', 'Social Sciences and History', 'Business',
          'Math and Statistics', 'Architecture', 'Physical Sciences',
          'Computer Science', 'Engineering']

y_offsets = {'Foreign Languages': 0.5, 'English': -0.5,
             'Communications\nand Journalism': 0.75,
             'Art and Performance': -0.25, 'Agriculture': 1.25,
             'Social Sciences and History': 0.25, 'Business': -0.75,
             'Math and Statistics': 0.75, 'Architecture': -0.75,
             'Computer Science': 0.75, 'Engineering': -0.25}

for rank, column in enumerate(majors):
    # Plot each line separately with its own color.
    column_rec_name = column.replace('\n', '_').replace(' ', '_').lower()

    line = plt.plot(gender_degree_data['Year'],
                    gender_degree_data[column],
                    lw=2.5,
                    color=color_sequence[rank])

    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    y_pos = np.array(gender_degree_data[column])[-1] - 0.5

    if column in y_offsets:
        y_pos += y_offsets[column]

    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    plt.text(2011.5, y_pos, column, fontsize=14, color=color_sequence[rank])

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
fig.suptitle('Percentage of Bachelor\'s degrees conferred to women in '
             'the U.S.A. by major (1970-2011)\n', fontsize=18, ha='center')

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')
plt.show()
