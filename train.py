import torch
import torch.nn as nn
import Model
import time
from DenoisingDataLoader import NoisyDataset, ToTensor
from torch.utils.data import DataLoader

MAX_EPOCH = 1000
BATCH_SIZE = 4
CHANNELS = 3
MODEL = Model.FeaturePyramidNetwork()


def save_model(model, epoch):
    torch.save(model, "./Checkpoints/" + model.name + "_{}.pth".format(epoch))


def train():
    img_transforms = ToTensor()
    dataset = NoisyDataset(csv_file="./TrainingSet.csv", noisy_dir="./TrainingSet/Images",
                           gt_dir="./TrainingSet/GroundTruths", transform=img_transforms)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    mse_loss = nn.MSELoss(reduction="sum")

    model = MODEL.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    epoch = 0

    for epoch in range(epoch, MAX_EPOCH):
        t0 = time.time()
        for data in data_loader:
            img, gt_img = data["image"], data["gt_img"]
            img = img.cuda().float()
            gt_img = gt_img.cuda().float()
            # ===================forward=====================
            output = model(img)
            loss = mse_loss(output, gt_img).cuda()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        t1 = time.time()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, MAX_EPOCH, loss.item()))
        print('Time elapsed: {:.4f}'.format(t1 - t0))
        if epoch % 5 == 4:
            # Save inference model at every 5th Epoch (Checkpoint is not implemented)
            save_model(model, epoch + 1)


if __name__ == '__main__':
    train()
