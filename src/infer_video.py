import cv2
import torch
import numpy as np
from PIL import Image
import typing as ty
import asyncio

import model
import utils_

prediction = None

@torch.no_grad()
async def make_prediction(image: np.ndarray, model_: torch.nn.Module) -> ty.Dict[str, torch.Tensor]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    image = Image.fromarray(image).convert("RGB")
    image = utils_.get_transform(train=False)(image, None)[0]
    global prediction
    prediction = model_(image.unsqueeze(0))[0]

async def infer(model_name: str, video_path: str, fps: int = 60) -> None:
    model_ = model.get_model(num_classes=7)
    model_.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    model_.eval()

    video = cv2.VideoCapture(video_path)

    if video.isOpened() == False:
        print("Whoops...")

    count = 0
    prediction_tasks = []
    while video.isOpened():

        ret, frame = video.read()
        if ret == True:

            if count % 10 == 0:
                task = asyncio.create_task(make_prediction(frame, model_))
                prediction_tasks += [task]
                pass
            
            if not prediction is None:
                for box in prediction["boxes"]:
                    x, y, width, height  = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
                    frame = cv2.rectangle(frame, (x, y), (x + width, y + height), (36,255,12), 1)
            
            cv2.imshow(f"FlyAI: {video_path}", frame)
        
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

            count += 1
        
        else: 
            break
    
    # for task in prediction_tasks:
        # await task

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(infer("models/model-02.pth", "data/video/bookstore/video0/video.mp4"))
    # infer("models/model-02.pth", "data/video/bookstore/video0/video.mp4")