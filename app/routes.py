import os
import requests

from fastapi import FastAPI
from pydantic import BaseModel
from banhammer import is_dublicate

from database import SessionLocal
from models import Video

#Создание экземпляра FastAPi
def get_application() -> FastAPI:
    application = FastAPI()
    return application

#Создание экземпляра приложения
app = get_application()

#Модель запроса
class videoLinkRequest(BaseModel):
    link: str

#Модель ответа
class videoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: str

async def upoadingVideo(link):
    
    response = requests.get(link)
    if response.status_code == 200:

        session = SessionLocal()
        query = session.query(Video.link).all()

        #Создание пути к папке videos
        VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'videos')
        
        #Создание папки videos в случае если папки нет
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        

        # Получаем имя файла из ссылки загруж. видео 
        file_path_client = os.path.join(VIDEOS_DIR, os.path.basename(link))
        # Получаем путь файла из бд
        
        file_path_db = os.path.join(VIDEOS_DIR, os.path.basename(query))



        # Сохраняем файлы
        with open(file_path_client, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with open(file_path_db, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        
        result = is_dublicate(file_path_client, file_path_db)
        return result

        
    else:
        print(response.status_code)



#Запрос на проверку видео
@app.post("/check-video-duplicate")
async def chek_video_duplicate(LinkVideo: videoLinkRequest):
    result = await upoadingVideo(LinkVideo.link)

    #В таком формате должен возвращаться ответ проверки видео
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
