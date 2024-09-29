from fastapi import FastAPI
from .routes import router
from .database import engine, Base

app = FastAPI()

# Подключение маршрутов
app.include_router(router)

