import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image
import librosa
import numpy as np
import time
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

app = Flask(__name__)


### Chroma features
def chroma(samples, sample_rate):
  chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
  chroma_stft = chroma_stft.ravel()
  chroma_stft = chroma_stft.tolist()
  chroma_stft.sort(reverse=True)
  return chroma_stft[:1137]

### Mel-Frequency Cepstral Coefficients
def mfcc(samples, sample_rate):
  mfcc = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
  mfcc = mfcc.ravel()
  mfcc = mfcc.tolist()
  mfcc.sort(reverse=True)
  return mfcc[:11809]

### Spectral flux
def spectral_flux(samples, sample_rate):
  onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
  onset_env = onset_env.ravel()
  onset_env = onset_env.tolist()
  onset_env.sort(reverse=True)
  return onset_env[:93]

### Zero-crossing rate
def zero_cross(samples, sample_rate):
  zero_cross = librosa.feature.zero_crossing_rate(samples)
  zero_cross = zero_cross.ravel()
  zero_cross = zero_cross.tolist()
  zero_cross.sort(reverse=True)
  # 
  return zero_cross[:93]

### Spectral-roll off
def spectral_roll(samples, sample_rate):
  S, phase = librosa.magphase(librosa.stft(samples))
  spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sample_rate)
  spectral_rolloff = spectral_rolloff.ravel()
  spectral_rolloff = spectral_rolloff.tolist()
  spectral_rolloff.sort(reverse=True)
  return spectral_rolloff[:93]

def feature_selection(cough_heavy_file):
  samples, sample_rate = librosa.load(cough_heavy_file)
  vec = []
  vec.append(chroma(samples, sample_rate))
  vec.append(mfcc(samples, sample_rate))
  vec.append(spectral_flux(samples, sample_rate))
  vec.append(zero_cross(samples, sample_rate))
  vec.append(spectral_roll(samples, sample_rate))

  ### flatten the vector
  vec = [element for sub_vec in vec for element in sub_vec]
  return vec

file_name = 'cough-heavy.wav'
file_path = 'C:/Users/vivek/Covid19_Cough-Detector/cough-heavy.wav'

vec = feature_selection(file_path)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(13225,5000)
        self.linear2 = nn.Linear(5000,2500)
        self.linear3 = nn.Linear(2500,1000)
        self.linear4 = nn.Linear(1000,500)
        self.linear5 = nn.Linear(500,100)
        self.linear6 = nn.Linear(100,10)
        self.linear7 = nn.Linear(10, 2)
    
    def forward(self,X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = F.relu(self.linear4(X))
        X = F.relu(self.linear5(X))
        X = F.relu(self.linear6(X))
        X = self.linear7(X)
        return F.log_softmax(X, dim=1)


mlp=MLP()
PATH='C:/Users/vivek/Covid19_Cough-Detector/fix_mlp_5.pth'

model_load= mlp.load_state_dict(torch.load(PATH))
mlp.eval()


# def get_file_path_and_save(request):
#     # Get the file from post request
#     f = request.files['file']   # Save the file to ./uploads
#     basepath = os.path.dirname(__file__)
#     file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#     fullfilepath=os.path.join(basepath, secure_filename(f.filename))
#     f.save(file_path)
#     print('filename is',f)
#     return f

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)

    audio_file = uploaded_file.filename
    vec_test = feature_selection(audio_file)

    vec_data = torch.LongTensor([vec_test])
    test_vec = Variable(vec_data).float()

    output = mlp(test_vec)
    predicted = torch.max(output,1)[1].tolist()
    if predicted[0] == 0:
      prediction = 'Negative'
    else:
      prediction = 'Positive'
    return render_template('result.html', prediction=prediction)



if __name__ == "__main__":    
    app.run(debug=True)
