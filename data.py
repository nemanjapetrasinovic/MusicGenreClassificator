import matplotlib.pyplot as plt

from settings import *
from os import path
import os
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import time
import matplotlib as plt
import cv2 as cv


def load_data(audio_dir):
    categories = []
    train_data = []
    test_data = []

    music_folders = os.listdir(audio_dir)
    for g in music_folders:  # g = genres
        path = os.path.join(audio_dir, g)
        if os.path.isdir(path):
            categories.append(g)

            cnt = 0
            files = os.listdir(path)
            for f in files:
                audio_file = os.path.join(path, f)
                if os.path.isfile(audio_file):
                    if cnt % 5 == 2:
                        test_data.append((audio_file, categories.index(g)))
                    else:
                        train_data.append((audio_file, categories.index(g)))
                    cnt += 1

    x_train = np.zeros((len(train_data), 128, 323, 3), dtype='uint8')  # detect spect_size
    y_train = np.zeros((len(train_data)), dtype='uint8')
    x_test = np.zeros((len(test_data), 128, 323, 3), dtype='uint8')
    y_test = np.zeros((len(test_data)), dtype='uint8')

    for i in range(len(train_data)):
        file, g = train_data[i]
        y_train[i] = g
        # spec = get_melspectrogram_db(file, genre=categories[g])
        img = cv.imread(file)
        x_train[i] = img

    for i in range(len(test_data)):
        file, g = test_data[i]
        y_test[i] = g
        # spec = get_melspectrogram_db(file, genre=categories[g])
        img = cv.imread(file)
        x_test[i] = img

    rand_train_idx = np.random.RandomState(seed=0).permutation(len(train_data))
    x_train = x_train[rand_train_idx]
    y_train = y_train[rand_train_idx]

    rand_test_idx = np.random.RandomState(seed=0).permutation(len(test_data))
    x_test = x_test[rand_test_idx]
    y_test = y_test[rand_test_idx]

    return categories, train_data, test_data, x_train, y_train, x_test, y_test

def get_melspectrogram_db(file_path, sr=22050, n_fft=2048, hop_length=2049, n_mels=128, fmin=20, fmax=8300, top_db=80, genre='None'):
    try:
        dst_folder = os.path.join('/content/gdrive/MyDrive/MusicGenreClassificator/dataset_500_png', genre)
        if (os.path.exists(dst_folder) == False):
            os.mkdir(dst_folder)

        dst = os.path.join(dst_folder, os.path.basename(file_path)[:-3] + 'png')
        print(dst)
        if (os.path.exists(dst) == True):
            return

        if (os.path.exists(file_path) == False):
            print("Path does not exists: " + file_path)

        audio,sr = librosa.load(file_path, sr=sr, mono=True)

        if audio.shape[0]<30*sr:
            audio=np.pad(audio,int(np.ceil((30*sr-audio.shape[0])/2)),mode='reflect')
        else:
            audio=audio[:30*sr]

        spec=librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)

        img = spec_to_image(spec_db)
        plt.pyplot.imsave(dst, img)

        return spec_db

    except Exception as ex:
            print (ex)
            return -1
            pass

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled