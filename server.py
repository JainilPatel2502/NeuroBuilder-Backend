from fastapi import FastAPI , UploadFile , File,Form , Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
import os
from pydantic import BaseModel 
from typing import List 
from Data.Datahandler import Datahandler
from ModelBuilder.Builder import model_builder
from utils.get_loss_fn import get_loss_fn
from utils.get_optimizer import get_optimizer
import torch.nn as nn
import torch
app = FastAPI()
UPLOAD = './Projects'
app.add_middleware(
CORSMiddleware,
allow_origins=['*'],
allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/upload')
async def upload(file:UploadFile=File(...),filename:str=Form(...)):
    with open(f'{UPLOAD}/{filename}.csv','wb') as f:
        f.write(await file.read())
    return JSONResponse({"ok":True})

@app.post('/get_project')
async def select():
    porjs = os.listdir(UPLOAD)
    names =[]
    for pro in porjs:
        name = pro.split('.')[0]
        names.append(name)
    return JSONResponse({"pojects":names})



class ProjectRequest(BaseModel):
    project_name: str
    type: str
    split: float
    batch_size: int

@app.post('/set_project')
async def set_project(data: ProjectRequest):
    project_name = data.project_name
    proj_type = data.type
    split = data.split
    batch_size = data.batch_size

    data_warehouse = Datahandler(project_name , proj_type, split , batch_size)
    return JSONResponse({'ok': True ,'data': data_warehouse.df.head(20).to_dict(orient="records"),'training':len(data_warehouse.trainloader),'testing':len(data_warehouse.testloader),'input_size':data_warehouse.traindataset.x.shape[1]})

class ModelData(BaseModel):
    actiavtionsPerLayer:List[str]
    epochs :int
    initializationPerLayer:List
    layers:int
    lossFn:str
    lr:float
    neuronsPerLayer:List[int]
    optimzer:str
    regularization:str
    input:int



@app.post("/build_model")
def model(data: ModelData):
    model = model_builder(data.input, data.model_dump())
    model_str = str(model) 
    print(model_str)

    return {
        "status": "Model received",
        "model": model_str
    }


class Train(BaseModel):
    data:ProjectRequest
    model_info:ModelData

@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    try:
        body = await websocket.receive_json()
        body = Train(**body)

        project_name = body.data.project_name
        proj_type = body.data.type
        split = body.data.split
        batch_size = body.data.batch_size

        data_warehouse = Datahandler(project_name, proj_type, split, batch_size)
        model_data = body.model_info
        model = model_builder(model_data.input, model_data.model_dump())

        lossfn = get_loss_fn(model_data.lossFn)
        if lossfn is None:
            await websocket.send_json({"error": f"Invalid loss function: {model_data.lossFn}"})
            return

        optimizer = get_optimizer(model.parameters(), model_data.optimzer, model_data.lr)
        if optimizer is None:
            await websocket.send_json({"error": f"Invalid optimizer: {model_data.optimzer}"})
            return
        train_losses = []
        test_losses = []

        for epoch in range(model_data.epochs):
            total_train_loss = 0
            for x, y in data_warehouse.trainloader:
                pred = model(x)
                y = y.long() if isinstance(lossfn, nn.CrossEntropyLoss) else y.float().view(-1, 1)
                loss = lossfn(pred, y)
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
                    total_test_loss += loss.item()

            train_losses.append(total_train_loss)
            test_losses.append(total_test_loss)

  
            await websocket.send_json({
                "epoch": epoch + 1,
                "train_loss": total_train_loss,
                "test_loss": total_test_loss,
                "message": f"Epoch [{epoch+1}/{model_data.epochs}], Train: {total_train_loss:.4f}, Test: {total_test_loss:.4f}"
            })

 
        await websocket.send_json({
            "message": "Training completed successfully",
            "train_losses": train_losses,
            "test_losses": test_losses
        })

        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected during training")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
