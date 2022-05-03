import librosa
import numpy as np
class AudioFeatureExtractor():
    def __init__(self,):
        super(AudioFeatureExtractor,self).__init__()

    def read_audio(self, audio_path, sr):
        audio, sr = librosa.load(audio_path,sr=sr)
        return audio, sr

    def get_features(self, X, sample_rate):

        stft = np.abs(librosa.stft(X))

        pitches, magnitudes = librosa.piptrack(X, sr = sample_rate, S = stft, fmin = 70, fmax = 400)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, 1].argmax()
            pitch.append(pitches[index, i])

        pitch_tuning_offset = librosa.pitch_tuning(pitches)
        pitchmean = np.mean(pitch)
        pitchstd = np.std(pitch)
        pitchmax = np.max(pitch)

        cent = librosa.feature.spectral_centroid(y = X, sr = sample_rate)
        cent = cent / np.sum(cent)
        meancent = np.mean(cent)
        stdcent = np.std(cent)
        maxcent = np.max(cent)

        flatness = np.mean(librosa.feature.spectral_flatness(y = X))

        mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
        mfccsstd = np.std(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
        mfccmax = np.max(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)

        chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis = 0)

        contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sample_rate).T, axis = 0)

        zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

        S, _ = librosa.magphase(stft)
        meanMagnitude = np.mean(S)
        stdMagnitude = np.std(S)
        maxMagnitude = np.max(S)

        rmse = librosa.feature.rms(S=S)[0]
        meanrms = np.mean(rmse)
        stdrms = np.std(rmse)
        maxrms = np.max(rmse)

        ext_features = np.array([
            pitch_tuning_offset, pitchmean, pitchmax, pitchstd,
            meancent, maxcent, stdcent,
            flatness, zerocr, 
            meanMagnitude, maxMagnitude, stdMagnitude, 
            meanrms, maxrms, stdrms
        ])

        ext_features = np.concatenate((ext_features, mfccs, mfccmax, mfccsstd, chroma, mel, contrast))
        
        return ext_features
