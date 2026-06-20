import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
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


@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket, db: Session = Depends(auth.get_db)):
    await websocket.accept()
    logger.info("New websocket connection accepted for training")
    try:
        body = await websocket.receive_json()
        body = Train(**body)

        try:
            current_user = auth.get_current_user(body.token, db)
        except HTTPException:
            logger.warning("Unauthorized websocket connection attempt")
            await websocket.send_json({"error": "Unauthorized"})
            await websocket.close()
            return

        logger.info("User %s initiated training for project '%s'", current_user.email, body.data.project_name)
        await run_training(websocket, body, current_user, db)

    except WebSocketDisconnect:
        logger.info("Client disconnected during training")
    except Exception as e:
        logger.error("Unexpected error in websocket_train: %s", str(e), exc_info=True)
        await websocket.send_json({"error": str(e)})
        await websocket.close()
