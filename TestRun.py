from Meow_Net import Meow_Net
from CosineAnnealing import CosineAnnealingScheduler
import numpy as np
import math
import matplotlib.pyplot as plt
from pydub import AudioSegment

# Import DataSets
file_list = []
label_list = []
for i in os.listdir('./dataset'):
    file_list.append(i)
    label_list.append(i.split('_')[2])
    
from pandas.core.frame import DataFrame
train= pd.DataFrame({'fname':file_list})

path = './dataset/'

import wave

def get_length(file):
    audio = wave.open(path+file)
    return audio.getnframes() / audio.getframerate()

from joblib import Parallel, delayed

with Parallel(n_jobs=10, verbose=1) as ex:
    lengths = ex(delayed(get_length)(e) for e in train.fname)
    
train['length'] = lengths   

def obtain_mfcc(file, features=128):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    return librosa.feature.mfcc(y, sr, n_mfcc=features)

def get_mfcc(file, n_mfcc=128, padding=None):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    if padding: mfcc = np.pad(mfcc, ((0, 0), (0, max(0, padding-mfcc.shape[1]))), 'constant')
    return mfcc.astype(np.float32)

from functools import partial

n_mfcc = 128
padding = 173
fun = partial(get_mfcc, n_mfcc=n_mfcc, padding=padding)

with Parallel(n_jobs=10, verbose=1) as ex:
    mfcc_data = ex(delayed(partial(fun))(e) for e in train.fname)
    
# Juntamos la data en un solo array y agregamos una dimension
mfcc_data = np.stack(mfcc_data)[..., None]

title = ['emission_context','cat_id','breed','sex','cat_owner_id','recording_session_and_vocalization_counter']

for j in range(len(title)):
    label_list = []
    for i in train['fname']:
        #print(j)
        label_list.append(i.split('_')[j])
    train[title[j]] = label_list
    
lbl2idx = {lbl:idx for idx,lbl in enumerate(train.breed.unique())}
idx2lbl = {idx:lbl for lbl,idx in lbl2idx.items()}
n_categories = len(lbl2idx)

train['y'] = train.breed.map(lbl2idx)

Ytrain = np.array(train['y'])
Ytrain=Ytrain.reshape(Ytrain.shape[0],1)

# change data type to float32
mfcc_data.astype('float32')
# Normalize pixel values to be between 0 and 1
mfcc_data = mfcc_data-mfcc_data.min() / (mfcc_data.max()-mfcc_data.min())

# Encoding
Ytrain = to_categorical(np.array(Ytrain[:, 0]))

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(mfcc_data, Ytrain, test_size=0.2, random_state=42)
inputs = np.zeros((1, x_train.shape[1], x_train.shape[2], x_train.shape[3]), dtype=np.float32)

def get_model():
    return Meow_Net(2)

input_shape = (None, 128, 173, 1)
model = get_model()
model.build(input_shape)
callbacks = [CosineAnnealingScheduler(T_max=100, eta_max=1e-2, eta_min=1e-4)]
model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, validation_split=0.25,callbacks=callbacks)


plt.style.use('seaborn')
plt.figure(figsize = (16,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Result',fontsize=20)
plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.legend(['accuracy','Validation_accuracy'], loc='lower right',fontsize=16)
plt.savefig('Training_Result.png')

score = model.evaluate(test_images, test_labels, verbose=0)
print('accuracy: ',score[1])
print('loss: ',score[0])