from pydantic import BaseModel
from datetime import datetime
import uuid
from typing import List, Optional

class videoLinkRequest(BaseModel):
    created: datetime
    uuid_video: uuid.UUID
    link: str
    is_duplicate: Optional[bool] = False
    duplicate_for: Optional[uuid.UUID] = None
    is_hard: Optional[bool] = False
    features: Optional[List[str]] = None

class VideoResponse(BaseModel):
    created: datetime
    uuid_video: uuid.UUID
    link: str
    is_duplicate: bool
    duplicate_for: Optional[uuid.UUID] = None
    features: Optional[List[str]] = None

    class Config:
        orm_mode = True
