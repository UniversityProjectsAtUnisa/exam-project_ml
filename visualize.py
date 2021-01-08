import librosa.display
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from matplotlib import pyplot as plt

SR = 16000


with open('./data/labeler.pk', 'rb') as f:
    labeler = pickle.load(f)


def visualize_element(elem, answer):
    lab = labeler.inverse_transform([np.argmax(answer)])[0]

    librosa.display.specshow(np.transpose(
        elem), x_axis='time', y_axis='mel', sr=SR, fmin=80, fmax=8000, hop_length=int(SR*0.01))
    plt.colorbar(format='%+2.0f dB')
    plt.title(lab)

    plt.show()
