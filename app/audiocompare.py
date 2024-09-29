import os
from moviepy.editor import VideoFileClip
import numpy as np
import librosa
from scipy.spatial.distance import cosine

# pip install moviepy librosa numpy scipy

def audiocompare(videopath1, videopath2, segment_duration=3.0, hop_duration=1.0, sr=22050):
    def extract_audio(video_file):
        video = VideoFileClip(video_file)
        audio_file = video_file.rsplit('.', 1)[0] + '.wav'
        video.audio.write_audiofile(audio_file)
        return audio_file

    def extract_mfcc(audio_file, sr=22050):
        y, sr = librosa.load(audio_file, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc.T

    def split_mfcc(mfcc, segment_duration, hop_duration, sr):
        segment_length = int(segment_duration * sr // 512)
        hop_length = int(hop_duration * sr // 512)
        segments = [mfcc[i:i + segment_length] for i in range(0, len(mfcc), hop_length)]
        return segments

    def compare_segments(segments1, segments2):
        min_length = min(len(segments1), len(segments2))
        segments1 = segments1[:min_length]
        segments2 = segments2[:min_length]

        similarities = []
        for seg1, seg2 in zip(segments1, segments2):
            if seg1.shape == seg2.shape:
                similarity = 1 - cosine(seg1.flatten(), seg2.flatten())
                similarities.append(similarity)

        return np.mean(similarities)

    audio1 = extract_audio(videopath1)
    audio2 = extract_audio(videopath2)

    mfcc1 = extract_mfcc(audio1, sr)
    mfcc2 = extract_mfcc(audio2, sr)

    segments1 = split_mfcc(mfcc1, segment_duration, hop_duration, sr)
    segments2 = split_mfcc(mfcc2, segment_duration, hop_duration, sr)

    result = compare_segments(segments1, segments2)

    # Удаление созданных аудиофайлов
    os.remove(audio1)
    os.remove(audio2)

    return result
