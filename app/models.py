from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

class Video(Base):
    __tablename__ = 'videos'

    created = Column(DateTime, nullable=False)
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    link = Column(String, nullable=False)
    is_duplicate = Column(Boolean, default=False)
    duplicate_for = Column(UUID(as_uuid=True), ForeignKey('videos.uuid'), nullable=True)
    is_hard = Column(Boolean, default=False)
    features = Column(ARRAY(String), nullable=True)

class ComplexVideo(Base):
    __tablename__ = 'complex_videos'

    created = Column(DateTime, nullable=False)
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    link = Column(String, nullable=False)
    is_duplicate = Column(Boolean, default=False)
    duplicate_for = Column(UUID(as_uuid=True), ForeignKey('complex_videos.uuid'), nullable=True)
    features = Column(ARRAY(String), nullable=True)
