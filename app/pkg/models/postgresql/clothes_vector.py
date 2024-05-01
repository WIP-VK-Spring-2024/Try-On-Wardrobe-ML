from sqlalchemy import Column, LargeBinary, UUID
from app.pkg.models.postgresql import Base

class ClothesVector(Base):
    __tablename__ = "clothes_vector"

    id = Column(UUID, primary_key=True)
    clothes_id = Column(UUID, nullable=False)
    tensor = Column(LargeBinary)
