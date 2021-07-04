import librosa 
import librosa.display
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, shutil, itertools, pickle, pandas as pd, seaborn as sn, math, time
from random import seed, random, randint
from scipy.spatial import distance
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf
from tqdm import tqdm
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from generate_triple_combinations import generate_pairs
from model import get_model
from scipy import spatial




# Read metadata file
metadata = pd.read_csv(os.getcwd()+"/BritishBirdSongDataset/birdsong_metadata.csv")
header = list(metadata.head())


# Get bird names
bird_name = metadata['english_cname'].values
u, f = np.unique(bird_name, return_counts=True)

# Take 5 unique bird names for this project
uniq_birds = list(u[4:10]) + list(u[12:15])
data_train = []
data_test = []
y_train = []
y_test = []
bird_name_dict = {}

# Get file_id corresponding to bird names
for i in range(len(uniq_birds)) :
    df = metadata[metadata['english_cname'] == uniq_birds[i]]
    df = df['file_id'].values
    df = df.tolist()
    data_train.append(df[0])
    y_train.append(i)
    bird_name_dict[i] = uniq_birds[i]
    data_test += df[1:]
    y_test += [i] * (len(df) - 1)




# Read audio files using librosa. Divide each audio clip into frames of 2 sec duration. They are more than 40 sec long clips. So, using frames
# we will get a lot of data.

# Read training data and split into frames
frames_train = []
frames_test = []
frame_len = 22050*2 # equivalent of 2 seconds
y_frames_train = []
y_frames_test = []


for i in tqdm(range(len(data_train))) :

    # Read audio
    curr = data_train[i]
    curr = os.getcwd() + "/BritishBirdSongDataset/songs/songs/xc" + str(curr) + ".flac"
    y, sr = librosa.load(curr)

    # Normalize time series
    y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1

    # Remove silence from the audio
    org_len = len(y)
    intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
    intervals = intervals.tolist()
    y = (y.flatten()).tolist()
    nonsilent_y = []

    for p,q in intervals :
        nonsilent_y = nonsilent_y + y[p:q+1] 

    y = np.array(nonsilent_y)
    final_len = len(y)
    sil = org_len - final_len


    # Divide audio into frames
    start = 0
    end = frame_len
    for j in range(0, len(y), int(frame_len*0.5)) :

        frame = y[j:j+frame_len]
        if len(frame) < frame_len :
            frame = frame.tolist() + [0]* (frame_len-len(frame))
        frame = np.array(frame)
        S = np.abs(librosa.stft(frame, n_fft=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        upper = ([x for x in range(len(freqs)) if freqs[x] >= 8000])[0]
        lower = ([x for x in range(len(freqs)) if freqs[x] <= 1000])[-1]

        freqs = freqs[lower:upper]
        S = S[lower:upper,:]
        if S.shape != (163, 345) :
            print(S.shape)
        assert S.shape == (163, 345)

        frames_train.append(S)
        y_frames_train.append(y_train[i])



# Read testing data and split into frames

for i in tqdm(range(len(data_test))) :

    # Read audio
    curr = data_test[i]
    curr = os.getcwd() + "/BritishBirdSongDataset/songs/songs/xc" + str(curr) + ".flac"
    y, sr = librosa.load(curr)
    dur = librosa.get_duration(y, sr)

    # Normalize time series
    y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1

    # Remove silence from the audio
    org_len = len(y)
    intervals = librosa.effects.split(y, top_db= 15, ref= np.max)
    intervals = intervals.tolist()
    y = (y.flatten()).tolist()
    nonsilent_y = []

    for p,q in intervals :
        nonsilent_y = nonsilent_y + y[p:q+1] 

    y = np.array(nonsilent_y)
    final_len = len(y)
    sil = org_len - final_len


    dur = librosa.get_duration(y, sr)
    start = 0
    end = frame_len
    for j in range(0, len(y), int(frame_len*0.5)) :
        frame = y[j:j+frame_len]
        if len(frame) < frame_len :
            frame = frame.tolist() + [0]* (frame_len-len(frame))
        frame = np.array(frame)

        S = np.abs(librosa.stft(frame, n_fft=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        upper = ([x for x in range(len(freqs)) if freqs[x] >= 8000])[0]
        lower = ([x for x in range(len(freqs)) if freqs[x] <= 1000])[-1]

        freqs = freqs[lower:upper]
        S = S[lower:upper,:]
        assert S.shape == (163, 345)

        frames_test.append(S)
        y_frames_test.append(y_test[i])


print("Training samples : ",  len(frames_train), len(y_frames_train), np.unique(y_frames_train, return_counts= True))
print("Testing samples : ",  len(frames_test), len(y_frames_test), np.unique(y_frames_test, return_counts= True))



# Convert all data to nd array
y_frames_train = np.array(y_frames_train)
y_frames_test = np.array(y_frames_test)

r,c = frames_train[0].shape
frames_train = np.array(frames_train)
frames_train = frames_train.reshape((len(frames_train), r, c))

frames_test = np.array(frames_test)
frames_test = frames_test.reshape((len(frames_test), r, c))

dataX = np.concatenate((frames_train, frames_test), axis=0)
datay = np.concatenate((y_frames_train, y_frames_test), axis=0)

frames_train, frames_test, y_frames_train, y_frames_test = train_test_split(dataX, datay, test_size=0.6)


# Write training and testing data into a pickle file
f = open(os.getcwd() + "/training_frames1.pkl", 'wb')
pickle.dump([frames_train, y_frames_train], f)
f.close()

f = open(os.getcwd() + "/testing_frames1.pkl", 'wb')
pickle.dump([frames_test, y_frames_test], f)
f.close()


# Read training and testing data from the pickle file
f = open(os.getcwd() + "/training_frames1.pkl", 'rb')
frames_train, y_frames_train = pickle.load(f)
f.close()

f = open(os.getcwd() + "/testing_frames1.pkl", 'rb')
frames_test, y_frames_test = pickle.load(f)
f.close()


# Standardize the data
mu = frames_train.mean()
std = frames_train.std()
frames_train = (frames_train-mu)/std
frames_test = (frames_test-mu)/std

print("Training samples : ",  frames_train.shape, len(y_frames_train), np.unique(y_frames_train, return_counts= True))
print("Testing samples : ", frames_test.shape, len(y_frames_test), np.unique(y_frames_test, return_counts= True))



# There are imbalanced classes. Repeat the data so that all classes have same number of samples.
u, f = np.unique(y_frames_train, return_counts= True)
frames_train1 = []
y_frames_train1 = []
maximum = max(f)
count = 0
for i in u :
    ind, = np.where(y_frames_train == i)
    ind = ind.tolist()
    while len(ind) < maximum :
        ind = ind + ind
    ind = ind[:maximum]
    temp = frames_train[ind]
    if count == 0 :
        frames_train1 = temp
        count += 1
    else :
        frames_train1 = np.concatenate((frames_train1, temp), axis= 0)

    y_frames_train1 += [i] * maximum

y_frames_train1 = np.array(y_frames_train1)





# Make pairs for siamese network
pos_X1, pos_X2, _, neg_X1 = generate_pairs(frames_train1, y_frames_train1, rand_samples= -1, pos_pair_size=1200, extra_data=[])

pos_X1 = pos_X1.astype(np.float16)
pos_X2 = pos_X2.astype(np.float16)
neg_X1 = neg_X1.astype(np.float16)

np.savez_compressed(os.getcwd()+'/training_siamese_frames', a=pos_X1, b= pos_X2, c= neg_X1)

data = np.load(os.getcwd()+'/training_siamese_frames.npz')        
pos_X1 = data['a']
pos_X2 = data['b']
neg_X1 = data['c']


print("Siamese pairs ", pos_X1.shape, pos_X2.shape, neg_X1.shape)




# Train the model
_,r,c = pos_X1.shape
encoder, model = get_model(c,r)

pos_X1 = pos_X2.transpose(0, 2, 1)
pos_X2 = pos_X2.transpose(0, 2, 1)
neg_X1 = neg_X1.transpose(0, 2, 1)

y = np.ones((len(pos_X1), 32*3))

# Fit model
mc = ModelCheckpoint(os.getcwd()+'/model_checkpoint.h5', 
                            save_weights_only=False, save_freq=10)


# Train the model
model.fit([pos_X1, pos_X2, neg_X1], y, epochs= 30, callbacks= [mc], batch_size= 256, verbose= 1)
# model.save(os.getcwd() + "/siamese.h5")
# encoder.save(os.getcwd() + "/encoder.h5")


# Load the model
model = load_model(os.getcwd() + "/model_checkpoint.h5", compile= False)
encoder = load_model(os.getcwd() + "/encoder.h5", compile= False)





# Create the confusion and similarity matrix
frames_test = frames_test.transpose(0,2,1)
frames_train1 = frames_train1.transpose(0,2,1)


sim_mat = np.zeros((len(frames_test), len(frames_test)))
ind = np.arange(0, len(frames_test), 1)
ind = [x for _,x in sorted(zip(y_frames_test, ind))]
frames_test = frames_test[ind]
y_frames_test = y_frames_test[ind]

encoded = encoder.predict(frames_test)

pred = []

for i in tqdm(range(len(frames_test))) :

    curr = frames_test[i:i+1]
    sim = encoder.predict(curr)
    sim = [1 - spatial.distance.cosine(sim[0], encoded[x]) for x in range(len(encoded))]
    sim = np.array(sim)

    sim[sim<0] = 0
    ind = np.copy(y_frames_test)
    
    ind = [x for _,x in sorted(zip(sim, ind))]
    ind = ind[::-1]
    ind = ind[:21]
    u, f = np.unique(ind, return_counts= True)
    best = u[np.argmax(f)]
    pred.append(best)
    sim_mat[i] = sim




conf_mat = confusion_matrix(y_frames_test, pred, normalize= 'true')
print("Accuracy on testset : ", np.trace(conf_mat)/np.sum(conf_mat))


u = np.unique(y_frames_test)


# Plot confusion matrix
conf_mat = pd.DataFrame(conf_mat, columns= [bird_name_dict[x] for x in u], index= [bird_name_dict[x] for x in u])
plt.figure(figsize = (20,15))
sn.heatmap(conf_mat, annot=True, annot_kws={"size": 10}, cmap='jet')
plt.tick_params(labelsize=8)
plt.xticks(rotation= 60)
plt.title("Bird Song classification")
# plt.show()
plt.savefig(os.getcwd()+"/ConfusionMatrix_test.png")
plt.close()


# Plot similarity matrix
sim_mat = pd.DataFrame(sim_mat, columns= [bird_name_dict[x] for x in y_frames_test], index= [bird_name_dict[x] for x in y_frames_test])
plt.figure(figsize = (20,15))
sn.heatmap(sim_mat, annot=False, annot_kws={"size": 12}, cmap='viridis')
plt.tick_params(labelsize=10)
plt.xticks(rotation= 90)
plt.title("Bird Song classification")
# plt.show()
plt.savefig(os.getcwd()+"/SimilarityMatrix_test.png")
plt.close()
