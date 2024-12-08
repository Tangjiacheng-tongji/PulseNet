from module import *
from meme import MemePools
from torchvision import models
from model.memenet import MemeNet

from collections import defaultdict

from params import test_loader,capacity, origin_interval
from tools import printr

interval = origin_interval

net = models.resnet18(pretrained=True)
net.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 7),
        )

memes = MemePools(capacity=capacity)
memenet = MemeNet(net, func = printr, device_ids=[0])

if __name__ == "__main__":
    memes.load('res18.json')

    kernels = memes.get_kernels()

    memenet.load_model("testres18(58.43).pth")

    #print("result:" + str(memenet.test_model(test_loader)))

    for i, (images, labels) in enumerate(test_loader):
        rec = ["anger","disgust","fear","happy","sad","surprised","normal"]
        idx = defaultdict(int)
        for s in range(len(images)):
            memenet.visualizing_v2(images[s], labels[s], kernels,\
                                   labels = rec, save_path="visualize/"+str(s))
            break
        break