import os
from typing import Dict

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

import models

UPLOAD = "./Vol/Projects"


def load_project_df(project_name: str, user: models.User, db: Session) -> pd.DataFrame:
    """Load a user's project CSV file as a pandas DataFrame."""
    project = (
        db.query(models.Project)
        .filter(models.Project.name == project_name, models.Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    path = f"{UPLOAD}/{project.filename}.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Project file not found")
    return pd.read_csv(path)


def column_type(series: pd.Series) -> str:
    """Return 'numeric' or 'categorical' based on the series dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def build_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return a dict mapping each column name to its detected type."""
    return {col: column_type(df[col]) for col in df.columns}
