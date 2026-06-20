import logging
import torch
import torch.nn as nn
from fastapi import WebSocket, HTTPException
from sqlalchemy.orm import Session

import models
from schemas import Train
from Data.Datahandler import Datahandler
from ModelBuilder.Builder import model_builder
from utils.get_loss_fn import get_loss_fn
from utils.get_optimizer import get_optimizer

logger = logging.getLogger("neurobuilder.trainer")


def regularization_penalty(model: nn.Module, regularization: str, weight: float) -> torch.Tensor:
    """Compute L1/L2 regularization penalty for model parameters."""
    if regularization == "L1":
        return weight * sum(param.abs().sum() for param in model.parameters())
    if regularization == "L2":
        return weight * sum((param ** 2).sum() for param in model.parameters())
    return torch.tensor(0.0)


async def run_training(websocket: WebSocket, body: Train, current_user: models.User, db: Session):
    """Execute the full training loop, streaming progress over the WebSocket."""
    project = (
        db.query(models.Project)
        .filter(models.Project.name == body.data.project_name, models.Project.user_id == current_user.id)
        .first()
    )
    if not project:
        logger.warning("Training aborted: Project '%s' not found for user %s", body.data.project_name, current_user.email)
        await websocket.send_json({"error": "Project not found"})
        await websocket.close()
        return

    proj_type = body.data.type
    split = body.data.split
    batch_size = body.data.batch_size

    logger.info("Starting training for user %s on project '%s' (epochs: %d, lr: %s, batch_size: %d)",
                current_user.email, project.name, body.model_info.epochs, body.model_info.lr, batch_size)

    data_warehouse = Datahandler(project.filename, proj_type, split, batch_size)
    model_data = body.model_info
    model = model_builder(model_data.input, model_data.model_dump())

    lossfn = get_loss_fn(model_data.lossFn)
    if lossfn is None:
        logger.error("Invalid loss function requested: %s", model_data.lossFn)
        await websocket.send_json({"error": f"Invalid loss function: {model_data.lossFn}"})
        return

    optimizer = get_optimizer(model.parameters(), model_data.optimizer, model_data.lr)
    if optimizer is None:
        logger.error("Invalid optimizer requested: %s", model_data.optimizer)
        await websocket.send_json({"error": f"Invalid optimizer: {model_data.optimizer}"})
        return

    train_losses = []
    test_losses = []
    reg_weight = model_data.regularizationStrength

    for epoch in range(model_data.epochs):
        total_train_loss = 0
        for x, y in data_warehouse.trainloader:
            pred = model(x)
            y = y.long() if isinstance(lossfn, nn.CrossEntropyLoss) else y.float().view(-1, 1)
            loss = lossfn(pred, y)
            loss = loss + regularization_penalty(model, model_data.regularization, reg_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        total_test_loss = 0
        with torch.no_grad():
            for x, y in data_warehouse.testloader:
                pred = model(x)
                y = y.long() if isinstance(lossfn, nn.CrossEntropyLoss) else y.float().view(-1, 1)
                loss = lossfn(pred, y)
                loss = loss + regularization_penalty(model, model_data.regularization, reg_weight)
                total_test_loss += loss.item()

        train_losses.append(total_train_loss)
        test_losses.append(total_test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == model_data.epochs - 1:
            logger.info("Epoch [%d/%d] - Train Loss: %.4f, Test Loss: %.4f", 
                        epoch + 1, model_data.epochs, total_train_loss, total_test_loss)

        await websocket.send_json({
            "epoch": epoch + 1,
            "train_loss": total_train_loss,
            "test_loss": total_test_loss,
            "message": f"Epoch [{epoch+1}/{model_data.epochs}], Train: {total_train_loss:.4f}, Test: {total_test_loss:.4f}",
        })

    logger.info("Training completed successfully for user %s on project '%s'", current_user.email, project.name)
    await websocket.send_json({
        "message": "Training completed successfully",
        "train_losses": train_losses,
        "test_losses": test_losses,
    })

    await websocket.close()
