import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

import auth
import models
from schemas import ModelData, Train
from ModelBuilder.Builder import model_builder
from training.trainer import run_training

logger = logging.getLogger("neurobuilder.model")
router = APIRouter(tags=["model"])


@router.post("/build_model")
def model(data: ModelData, current_user: models.User = Depends(auth.get_current_user)):
    try:
        built_model = model_builder(data.input, data.model_dump())
        model_str = str(built_model)
        logger.info("Successfully built model for user %s: \n%s", current_user.email, model_str)

        return {
            "status": "Model received",
            "model": model_str,
        }
    except Exception as e:
        logger.error("Error building model for user %s: %s", current_user.email, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/train")
async def train_model(
    body: Train,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(auth.get_db),
):
    logger.info("User %s initiated training for project '%s'", current_user.email, body.data.project_name)
    return StreamingResponse(
        run_training(body, current_user, db),
        media_type="text/event-stream"
    )
