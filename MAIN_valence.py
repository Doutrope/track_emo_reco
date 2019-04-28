


            ##########################################
            #       Modelize Valence                 #
            #             with LSTMs                 #
            ##########################################


# PREDICT VALENCE MODULATION WITH MANY-TO-MANY LSTM & MEAN VALENCE 
# WITH MANY-TO-ONE LSTM
            
# see : https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed, Bidirectional
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
emo = np.take(valence,np.array(list(range(60))) + 1,1)

y = emo

del [arousal,valence] 

# ==============
# IMPORT CHROMAS
# ==============

# input MFCC dimensions
timesteps, features, n = np.load('data\\DEAM_chromas\\' + random.choice(os.listdir('data\\DEAM_chromas\\'))).shape

# Import MFCCs
printcounter = 0
gpecounter = 1
nb_files = len(os.listdir('data\\DEAM_chromas\\'))
i=0
index = np.array([])
x = np.zeros((nb_files,timesteps,features))
with progressbar.ProgressBar(max_value=len(os.listdir('data\\DEAM_chromas\\'))) as bar:
    for file in os.listdir('data\\DEAM_chromas\\'):
        tmp = np.load('data\\DEAM_chromas\\' + file)
        tmp = np.reshape(tmp,(60,12))
        x[i,:,:] = tmp
        index = np.append(index,file)
        i+=1
        bar.update(i)

y = np.reshape(y,(nb_files,timesteps,1))
ymean = np.mean(y,axis = 1)
# ======================
# TRAIN - TEST - RESHAPE
# ======================

# Sample train test
indices = np.random.permutation(x.shape[0])
training_idx, test_idx = indices[:int(x.shape[0]*0.7)], indices[int(x.shape[0]*0.7):]
x_train, x_test = x[training_idx,:], x[test_idx,:]
y_train, y_test = y[training_idx], y[test_idx]
ymean_train, ymean_test = ymean[training_idx], ymean[test_idx]
del [tmp]

# Reshape arrays
#x_train = np.reshape(x_train,(x_train.shape[0]*x_train.shape[1],
#                              x_train.shape[2],
#                              x_train.shape[3]))
#x_test = np.reshape(x_test,(x_test.shape[0]*x_test.shape[1],
#                            x_test.shape[2],
#                            x_test.shape[3]))
#y_train = np.reshape(y_train,(y_train.shape[0]*y_train.shape[1]))
#y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]))

# Once train and test separated by song, reshuffle x and y and sample
#indices_train = np.random.permutation(x_train.shape[0])
#sample_train_idx = indices_train
#x_train, y_train = x_train[sample_train_idx,:], y_train[sample_train_idx]

#indices_test = np.random.permutation(x_test.shape[0])
#sample_test_idx = indices_test
#x_test, y_test = x_test[sample_test_idx,:], y_test[sample_test_idx]

# ====================
# KERAS INITIALISATION
# ====================

# ConvNets accepts 2 data formats according to Keras backend :
#   - If channel last, backend takes data as [rows, cols, channels]
#   - If channel first, backend takes data as [channels, rows, cols]

#if K.image_data_format() == 'channels_first': 
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#    input_shape = (img_rows, img_cols, 1)

# ==============
# NORMALISATIONS
# ==============
    
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
ymean_train = ymean_train.astype('float32')
ymean_test = ymean_test.astype('float32')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('ymean_train shape:', ymean_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# =================
# LSTM MANY-TO-MANY
# =================

batch_size = 128
epochs = 100
score_many = list() 
for neur in [5,10]:

    n_neurons = neur
    tmp_score = list()
    
    for i in range(10):

        model_many = Sequential()
        model_many.add(LSTM(n_neurons, input_shape=(timesteps, features), return_sequences=True))
        model_many.add(TimeDistributed(Dense(1)))
        model_many.compile(loss='mean_squared_error', optimizer='adam')
        
        ES = [EarlyStopping(monitor='val_loss',patience=5,verbose=2,mode='auto')]
        model_many.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks=ES)
        
        tmp_score.append(model_many.evaluate(x_test, y_test, verbose=1))

    score_many.append(np.mean(tmp_score))

# ===============================
# BIDIRECTIONAL LSTM MANY-TO-MANY
# ===============================

# (enhanced version of LSTM where all inputs are known for prediction, ie 
#   it uses also the future to predict the past)

batch_size = 128
epochs = 100
score_bi_many = list() 
for neur in [5,10]:

    n_neurons = neur
    tmp_score = list()
    
    for i in range(10):
        
        model_bi_many = Sequential()
        model_bi_many.add(Bidirectional(LSTM(n_neurons, return_sequences=True), input_shape=(timesteps, features)))
        model_bi_many.add(TimeDistributed(Dense(1)))
        model_bi_many.compile(loss='mean_squared_error', optimizer='adam')
        
        ES = [EarlyStopping(monitor='val_loss',patience=5,verbose=2,mode='auto')]
        model_bi_many.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks=ES)
        
        tmp_score.append(model_bi_many.evaluate(x_test, y_test, verbose=1))

    score_bi_many.append(np.mean(tmp_score))

# ================
# LSTM MANY-TO-ONE
# ================

batch_size = 128
epochs = 100
n_neurons = 50

tmp_score = list()

model_one = Sequential()
model_one.add(LSTM(50, input_shape=(timesteps, features)))
model_one.add(Dense(1))
model_one.compile(loss='mae', optimizer='adam')

ES = [EarlyStopping(monitor='val_loss',patience=5,verbose=2,mode='auto')]
model_one.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=ES)

model_one.evaluate(x_test, ymean_test, verbose=1)



score_one = model_one.evaluate(x_test, ymean_test, verbose=1)
score_many = model_many.evaluate(x_test, y_test, verbose=1)
score_bi_many = model_bi_many.evaluate(x_test, y_test, verbose=1)
print('Test loss one:', score_one[0])
print('Test MSE one :', score_one[1])
print('Test loss many:', score_many[0])
print('Test RMSE many :', np.sqrt(score_many[1]))
print('Test loss bi many:', score_bi_many[0])
print('Test RMSE bi many :', np.sqrt(score_bi_many[1]))


# ===============================
# TRAIN BEST MODEL ON ALL SAMPLES
# ===============================

model_many = Sequential()
model_many.add(LSTM(5, input_shape=(timesteps, features), return_sequences=True))
model_many.add(TimeDistributed(Dense(1)))
model_many.compile(loss='mean_squared_error', optimizer='adam')

ES = [EarlyStopping(monitor='loss',patience=5,verbose=2,mode='auto')]
history = model_many.fit(x, y,
                         batch_size=128,
                         epochs=100,
                         verbose=1,
                         callbacks=ES)

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

prediction_one = model_one.predict(x_test)
prediction_many = model_many.predict(x_test)

# Look at mean emo by track from many to many lstm
mean_from_mtom = np.mean(prediction_many,axis=1)

print('Many to one RMSE : ',np.sqrt(np.mean((prediction_one-ymean_test)**2)),
      '\nMany to many meaned RMSE : ',np.sqrt(np.mean((mean_from_mtom-ymean_test)**2)))

# ==========
# SAVE MODEL
# ==========

# Save model and weightings
model_many.save('C:\\Users\\sacha\\Documents\\Projets\\DeepLearning\\keras-EmoReco\\models\\model_valen_testRMSE.h5')

# Save model architecture without weightings in json format
with open('C:\\Users\\sacha\\Documents\\Projets\\DeepLearning\\keras-EmoReco\\models\\model_valen_testRMSE_architecture.json', 'w') as f:  
    f.write(model_many.to_json())  
    
# Load model and weightings
model = keras.models.load_model('models/model_valen_testRMSE.h5')

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
        
        test_chroma = AudioUtils.compute_multi_chroma(os.path.join('data','deezer_wav',file),limit = np.inf)
        
        # Norm by train tracks
        #test_mfcc = (test_mfcc - x_train_min)/x_train_max
        # Norm by track itself
        #test_mfcc = (test_mfcc - np.mean(test_mfcc))/np.std(test_mfcc)
        # Norm each mfcc
        #for j in range(test_mfcc.shape[0]):
        #    test_mfcc[j] = (test_mfcc[j] - np.mean(test_mfcc[j]))/(np.std(test_mfcc[j])+1e-7) 
        
        test_chroma = test_chroma.reshape(1,test_chroma.shape[0],test_chroma.shape[1])
        
        test_pred = model.predict(test_chroma)#.T
        
        test_pred = np.reshape(test_pred,(1,60))
        
        test_pred[0] = smoothize(test_pred[0],3)
        
        test_pred = np.concatenate((np.array([[file]]),
                                    test_pred),axis=1)
        
        if i == 0:
            songs_predictions = test_pred
        else:
            songs_predictions = np.concatenate((songs_predictions,test_pred),axis = 0)
        
        i += 1
        bar.update(i)
        
test_pred = songs_predictions[:,1:]
test_pred = pd.DataFrame(songs_predictions[:,1:],
                         index=songs_predictions[:,0],
                         dtype = 'float32')
#test_pred.columns = np.arange(15.75, 23, 0.5).astype(str)

#test_pred.to_csv(os.path.join('data','DeezerSongs_arousal.csv'),sep=',')
#test_pred.to_csv(os.path.join('data','DeezerSongs_arousal.csv'),sep=',')
#test_pred = pd.read_csv(os.path.join('data','DeezerSongs_arousal.csv'),index_col = 0)

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
#time_ser.to_csv(os.path.join('data','DeezerSongs_arousal_ts.csv'),sep=',')

#time_ser = pd.read_csv(os.path.join('data','DeezerSongs_arousal_ts.csv'),index_col=0)

#time_ser['seconds'] = time_ser.index

mini = min(time_ser.min().tolist())
maxi = max(time_ser.max().tolist())

valen_preds = {}
for column in time_ser:
    valen_preds[column] = [round(100*(x-mini)/(maxi-mini),0) for x in time_ser[column].tolist()]

import json
with open('data/' + 'valen_predictions.json', 'w') as fp:
    json.dump(valen_preds, fp)

# COMPUTE JS CMDS AS SHLAG
 
cmds = ['<option value="'+x+'">'+x.split('.')[0]+'</option>' for x in valen_preds.keys()]

with open('valen_js.txt', mode='wt', encoding='utf-8') as f:
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
