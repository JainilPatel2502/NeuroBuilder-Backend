import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

import auth
import models
from schemas import ProjectRequest
from helpers import UPLOAD
from Data.Datahandler import Datahandler

logger = logging.getLogger("neurobuilder.projects")
router = APIRouter(tags=["projects"])


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    filename: str = Form(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    try:
        actual_filename = f"user{current_user.id}_{filename}"
        with open(f"{UPLOAD}/{actual_filename}.csv", "wb") as f:
            f.write(await file.read())

        # Check if project already exists for user
        existing = (
            db.query(models.Project)
            .filter(models.Project.name == filename, models.Project.user_id == current_user.id)
            .first()
        )
        if existing:
            existing.filename = actual_filename
            logger.info("Updated existing project '%s' for user %s", filename, current_user.email)
        else:
            project = models.Project(name=filename, filename=actual_filename, user_id=current_user.id)
            db.add(project)
            logger.info("Created new project '%s' for user %s", filename, current_user.email)
        db.commit()
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error("Error uploading file %s for user %s: %s", filename, current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/get_project")
async def select(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    try:
        projects = db.query(models.Project).filter(models.Project.user_id == current_user.id).all()
        names = [p.name for p in projects]
        logger.info("Retrieved %d projects for user %s", len(names), current_user.email)
        return JSONResponse({"projects": names})
    except Exception as e:
        logger.error("Error retrieving projects for user %s: %s", current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/set_project")
async def set_project(
    data: ProjectRequest,
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
            logger.warning("Project '%s' not found for user %s", data.project_name, current_user.email)
            raise HTTPException(status_code=404, detail="Project not found")

        proj_type = data.type
        split = data.split
        batch_size = data.batch_size

        logger.info("Loading project '%s' for user %s with type=%s, split=%s, batch_size=%s", 
                    data.project_name, current_user.email, proj_type, split, batch_size)
        
        data_warehouse = Datahandler(project.filename, proj_type, split, batch_size)
        return JSONResponse({
            "ok": True,
            "data": data_warehouse.df.head(20).to_dict(orient="records"),
            "training": len(data_warehouse.trainloader),
            "testing": len(data_warehouse.testloader),
            "input_size": data_warehouse.traindataset.x.shape[1],
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error loading project '%s' for user %s: %s", data.project_name, current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
