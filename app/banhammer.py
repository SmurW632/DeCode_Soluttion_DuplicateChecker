import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

from audiocompare import  audiocompare
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(pretrained=True).cuda()
model.eval()

# Функция для извлечения признаков из видео
def extract_features(video_path, model, frames_count=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    features = []

    # Предобработка изображений
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Grayscale()
    ])

    with torch.no_grad():
        while cap.isOpened() and frame_count < frames_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Предобработка изображения
            tensor_frame = preprocess(frame).cuda()
            tensor_frame = tensor_frame.unsqueeze(0)  # Добавляем размер пакета

            # Извлечение признаков
            feature = model(tensor_frame).cuda()
            features.append(feature.cpu())#сохранять признаки

            frame_count += 1

    cap.release()

    # Усреднение признаков для получения одного вектора
    return torch.mean(torch.stack(features), dim=0).squeeze().numpy()


# Функция для сравнения двух видео
def compare_videos(video_path1, video_path2):
    # Загружаем предобученную модель ResNet

    # Извлекаем признаки из видео
    features1 = extract_features(video_path1, model)
    features2 = extract_features(video_path2, model)
    
    # Вычисление косинусного сходства
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

    return similarity

import json
import os

def is_dublicate(video_path1, video_path2):
    similarity = compare_videos(video_path1, video_path2)
    print(similarity)
    truepath = os.path.basename(video_path2)
    
    if 0.8 < similarity < 0.9:
        audiocomparing = audiocompare(video_path1, video_path2)
        if audiocomparing < 0.8:
            return False
        else:
            return json.dumps({"is_duplcate": True, "duplicate_for": truepath})
    elif similarity > 0.9:
        return json.dumps({"is_duplicate": True, "duplicate_for": truepath})
    else:
        return False

