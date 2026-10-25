#waveform
#fft -> spectrum
#stft -> spectrogram
#MFCC

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
file = 'audio/audio_for_ml.wav'
#waveform , signal->1D array -values = sr *duration of song in sec, each value will be the amplitude of the song, sr->sampling rate
signal, sr =librosa.load(file, sr=22050)
librosa.display.waveshow(signal, sr=sr)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()
#fft
fft =np.fft.fft(signal)
#abs of the complex values to get the magnitude which gives contribution of each frequency to the signal        
magnitude = np.abs(fft)
#linspace gives evenly spaced values between 0 and sr, len(magnitude) gives the number of values to be generated
frequencies = np.linspace(0, sr, len(magnitude))  
#it is a symmetric function, so we only need the first half of the values
left_frequency = frequencies[:len(frequencies)//2]
left_magnitude = magnitude[:len(magnitude)//2]  
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

#stft -> spectogram
#n_fft -> number of samples per fft, hop_length -> number of samples between successive frames 
stft =librosa.core.stft(signal, n_fft=2048, hop_length=512)
#abs of the complex values to get the magnitude which gives contribution of each frequency to the signal
spectrogram = np.abs(stft)
#convert amplitude to decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)
#specshow heatmap like representation of the spectrogram, sr->sampling rate, hop_length->number of samples between successive frames
# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=512)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()

#MFCC
#n_mfcc -?no of coeffs of MFCCs to be extracted, n_fft -> number of samples per fft, hop_length -> number of samples between successive frames
MFCCs = librosa.feature.mfcc(y=signal,sr=sr, n_fft= 2048,hop_length =512, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=512)
plt.xlabel('Time')
plt.ylabel('MFCC')
plt.colorbar()
plt.show()