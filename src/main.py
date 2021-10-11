import torch.utils.data
import torch
from PIL import Image

import logging
import tqdm

import dataset
import model
import utils_

import utils

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(TqdmLoggingHandler())

@torch.no_grad()
def eval(model: torch.nn.Module, image_input: torch.Tensor, image: Image) -> None:
    model.eval()
    pred = model(image_input.unsqueeze(0))[0]
    utils_.plot_bboxes(image, pred)

def train(model_: torch.nn.Module, data_loader: torch.utils.data.DataLoader, epochs: int = 10):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    
    model_.to(device)

    params = [p for p in model_.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.005, momentum=.9, weight_decay=.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

    model_.train()
    for epoch in range(epochs):

        for i_batch, data in enumerate(tqdm.tqdm(data_loader, total=len(data_loader))):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model_(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr_scheduler.step()

            # if i_batch % 10 == 0:
                # log.info(f"Epoch {epoch}: {round(losses.cpu().detach().item(), 2)}")
        
        print(f"Epoch {epoch}: {losses}")