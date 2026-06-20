import logging
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

import auth
import models
from schemas import DataPreviewRequest, DataStatsRequest, DataDeleteRequest
from helpers import UPLOAD, load_project_df, column_type, build_column_types

logger = logging.getLogger("neurobuilder.data")
router = APIRouter(prefix="/data", tags=["data"])


@router.post("/delete")
async def data_delete(
    data: DataDeleteRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    try:
        project = (
            db.query(models.Project)
            .filter(models.Project.name == data.project_name, models.Project.user_id == current_user.id)
            .first()
        )
        if not project:
            logger.warning("Failed to delete: Project '%s' not found for user %s", data.project_name, current_user.email)
            raise HTTPException(status_code=404, detail="Project not found")

        path = f"{UPLOAD}/{project.filename}.csv"
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted file %s for user %s", path, current_user.email)

        db.delete(project)
        db.commit()
        logger.info("Deleted project record '%s' for user %s", data.project_name, current_user.email)
        return JSONResponse({"ok": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting project '%s' for user %s: %s", data.project_name, current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/preview")
async def data_preview(
    data: DataPreviewRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    try:
        df = load_project_df(data.project_name, current_user, db)
        preview = df.head(data.limit).to_dict(orient="records")
        logger.info("Generated preview for project '%s', user %s (rows: %d)", data.project_name, current_user.email, len(preview))
        return JSONResponse({
            "ok": True,
            "data": preview,
            "columns": list(df.columns),
            "column_types": build_column_types(df),
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating preview for project '%s', user %s: %s", data.project_name, current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/stats")
async def data_stats(
    data: DataStatsRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    try:
        df = load_project_df(data.project_name, current_user, db)
        if data.column not in df.columns:
            logger.warning("Stats request failed: Column '%s' not found in project '%s'", data.column, data.project_name)
            raise HTTPException(status_code=400, detail="Column not found")

        series = df[data.column]
        col_type = data.force_type if data.force_type else column_type(series)

        if col_type == "numeric":
            clean = pd.to_numeric(series, errors="coerce")
            clean = clean.dropna()
            if clean.empty:
                logger.info("Generated empty numeric stats for column '%s', project '%s'", data.column, data.project_name)
                return JSONResponse({"ok": True, "type": "numeric", "stats": {}, "histogram": {"bins": [], "counts": []}})

            counts, bin_edges = np.histogram(clean.values, bins=data.bins)
            stats = {
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "std": float(clean.std(ddof=0)),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "missing": int(series.isna().sum()),
            }
            logger.info("Generated numeric stats for column '%s', project '%s'", data.column, data.project_name)
            return JSONResponse({
                "ok": True,
                "type": "numeric",
                "stats": stats,
                "histogram": {
                    "bins": bin_edges.tolist(),
                    "counts": counts.tolist(),
                },
            })

        counts = series.fillna("(missing)").astype(str).value_counts()
        logger.info("Generated categorical stats for column '%s', project '%s'", data.column, data.project_name)
        return JSONResponse({
            "ok": True,
            "type": "categorical",
            "counts": {
                "labels": counts.index.tolist(),
                "values": counts.values.tolist(),
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating stats for column '%s', project '%s': %s", data.column, data.project_name, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
