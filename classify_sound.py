import sys
import numpy as np
import librosa
from essentia.standard import *
import numpy as np
import muda
import jams
import librosa
import keras
from keras.models import load_model


def mel128_full(audio_signal,Fs = 44100):  # MEL SPECTROGRAM, 128 bins over entire clip

    spectrum = Spectrum()
    w = Windowing(type = 'hann')


    length = int(round(0.02321995464*Fs)) # number of samples (1024) corresponding to a window size of 23 ms
    melbands = []
    mel = MelBands(highFrequencyBound = Fs/2,numberBands = 128, log = True,normalize = 'unit_max',type = 'magnitude',sampleRate = Fs )

    for frame in FrameGenerator(audio_signal, frameSize = length, hopSize = length,startFromZero=True,validFrameThresholdRatio=1):

        frame = np.concatenate((frame,np.zeros((1024))))
        frame = frame.astype('float32')

#         spec_zeropad = np.concatenate(((spectrum(w(frame))),np.zeros(1)))
#         spec_zeropad = spec_zeropad.astype('float32')
#        mel_specs = mel(spec_zeropad)

        mel_specs = mel(spectrum(w(frame)))
        melbands.append(mel_specs)

    melbands= essentia.array(melbands).T
    return melbands


labels = [ 'air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

filename = str(sys.argv[1]) # get the filename from the terminal input



model = load_model('weights_final_test.hdf5')



# load the soundfile
y,sr = librosa.load(filename,sr=44100,mono=True)

# calculate its mel bands
X = mel128_full(y)



# reshape it such that it is compatible with our convnet
X_test = np.zeros((X.shape[1]-128,X.shape[0],X.shape[0]))

for ii in range(X_test.shape[0]):
    X_test[ii,:,:] = X[:,ii:ii+128]
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2],1))


# Make our prediction
y_pred = model.predict(X_test)

label_pred = np.zeros((10,1))


# store every prediction made by the network for each frame, output the majority prediction
for ii in range(0,X_test.shape[0]):
    label_tmp = np.where(y_pred[ii,:]==np.max(y_pred[ii,:]))
    #print(np.where(y_pred[ii,:]==np.max(y_pred[ii,:])))
    label_pred[label_tmp[0]]+=1
prediction_final = np.where(label_pred==np.max(label_pred))[0]
print('1st predicted class : ' , labels[int(prediction_final)])
prediction_second = np.where(label_pred==np.sort(np.resize(label_pred,label_pred.shape[0]))[-2])[0][0]
print('2nd predicted class : ',labels[int(prediction_second)])
print('')
print('prediction probability per class: ')
print('')
for jj in range(len(labels)):
    print(labels[jj], (label_pred[jj]/np.sum(label_pred))*100,' % ')
