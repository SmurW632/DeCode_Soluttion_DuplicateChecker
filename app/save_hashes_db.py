import hashlib
import cv2
import torch
import numpy as np
import os
import multiprocessing
import logging
import uuid
from torchvision.transforms import transforms
from torchvision import models
from models import Video, ComplexVideo


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


DATABASE_URL = "postgresql://myuser:1@localhost/upload_db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    import models  # Импортируем модели перед созданием таблиц
    Base.metadata.create_all(bind=engine)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_model():
    # Load the model on GPU
    model = models.mobilenet_v3_large(pretrained=True).cuda()
    model.eval()
    return model

def hash_feature(feature):
    # Преобразуем объект в байты
    feature_bytes = feature.tobytes()
    # Вычисляем MD5 хеш
    md5_hash = hashlib.md5(feature_bytes).hexdigest()
    return md5_hash

def send_to_database(video_name, features):
    hashed_features = [hash_feature(feature) for feature in features]
    #hashed_features - это хешированные признаки
    
    # Извлекаем UUID из имени файла (предполагается, что название файла соответствует UUID)
    video_uuid = uuid.UUID(video_name.split('.')[0])

    session = SessionLocal()
    try:
        # Ищем запись в таблице Video с соответствующим UUID
        video_record = session.query(Video).filter(Video.uuid == video_uuid).first()
        if video_record:
            # Обновляем поле features
            video_record.features = hashed_features
            session.commit()
            logging.info(f"Updated database for video {video_name}")
        else:
            logging.warning(f"No record found in database for video {video_name}")
    except Exception as e:
        session.rollback()
        logging.error(f"Exception occurred while updating database: {str(e)}")
    finally:
        session.close()

    logging.info(f"Sending to database: {video_name}, Features: {features.shape}")

def extract_features(video_path, model, frames_count=30, batch_size=10):
    logging.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None

    frame_count = 0
    features = []

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        batch_frames = []
        while cap.isOpened() and frame_count < frames_count:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame from video: {video_path}")
                break

            logging.info(f"Processing frame {frame_count} of video {video_path}")

            tensor_frame = preprocess(frame)
            batch_frames.append(tensor_frame)

            if len(batch_frames) == batch_size or frame_count == frames_count - 1:
                batch_tensor = torch.stack(batch_frames).cuda()
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu())  # Transfer back to CPU and store features
                batch_frames = []

            frame_count += 1

    cap.release()
    if len(features) == 0:
        return None

    return torch.mean(torch.cat(features), dim=0).squeeze().numpy()

def process_video(video_path, index):
    try:
        model = load_model()
        features = extract_features(video_path, model)
        if features is not None:
            video_name = os.path.basename(video_path)
            send_to_database(video_name, features)
            logging.info(f"Processed video {index} with shape {features.shape}")
        else:
            logging.warning(f"Failed to process video {index}")
    except Exception as e:
        logging.error(f"Exception occurred while processing video {index}: {str(e)}")

if __name__ == '__main__':
    init_db()
    video_folder_path = r"d:\Downloads\DS_dup_check\train_dataset_train_data_yappy\train_data_yappy\train_dataset"
    video_files = os.listdir(video_folder_path)
    video_paths = [os.path.join(video_folder_path, video_file) for video_file in video_files]

    logging.info(f"Found {len(video_files)} video files.")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.starmap(process_video, [(video_path, index) for index, video_path in enumerate(video_paths)])
    pool.close()
    pool.join()

    logging.info("All videos processed")
