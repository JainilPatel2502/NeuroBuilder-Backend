from pydantic import BaseModel
from typing import List, Optional


class UserCreate(BaseModel):
    full_name: str
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class ProjectRequest(BaseModel):
    project_name: str
    type: str
    split: float
    batch_size: int


class DataPreviewRequest(BaseModel):
    project_name: str
    limit: int = 10000


class DataStatsRequest(BaseModel):
    project_name: str
    column: str
    bins: int = 10
    force_type: Optional[str] = None


class DataDeleteRequest(BaseModel):
    project_name: str


class ModelData(BaseModel):
    activationsPerLayer: List[str]
    epochs: int
    initializationPerLayer: List
    layers: int
    lossFn: str
    lr: float
    neuronsPerLayer: List[int]
    optimizer: str
    regularization: str
    regularizationStrength: float
    input: int


class Train(BaseModel):
    data: ProjectRequest
    model_info: ModelData
    token: Optional[str] = None

