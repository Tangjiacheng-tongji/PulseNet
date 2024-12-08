import time
from torchvision import models
from module import *
from meme import MemePools
from tools import Zscore,get_layers
from model.memenet import MemeNet

from params import extract_iter, train_loader, test_loader,\
    targets, target_channels, target_stride, meme_sizes,\
    num_channels, num_labels, capacity, \
    momentum, origin_interval, prune_iter, p, keep_channel
from tools import printr

interval = origin_interval

net = models.densenet121(pretrained=True)
net.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 196),
        )

memes = MemePools(capacity=capacity)

if __name__ == "__main__":
    memenet = MemeNet(net, func=printr, device_ids=[0])

    targets, target_channels, target_stride, meme_sizes = memenet.update_name(targets,
                                                  target_channels, target_stride, meme_sizes)
    zscore = Zscore(target_channels)

    memenet.load_prototype("model_85.3.pkl", "parts")
    memenet.demonstrate()

    memes.load('cifar10.json')
    zscore.load("zscore_cifar10.json")
    
    kernels = memes.get_kernels()
    memenet.add_memes(kernels, zscore_ED_mask, zscore.zscore)


    memenet.making_classifier((), 196)
    #memenet.formal_train(train_loader, test_loader, epochs = 1,  vis_iter = 200, lr=1e-2, test = False,  mode = 1)
    memenet.formal_train(train_loader, test_loader, epochs=25, vis_iter=50, lr=1e-3, test=False, mode=1,
                         formal_lr_step_size=1000)

    acc = memenet.test_model(test_loader)
    msg = "{}".format(acc)
    print(msg)
    memenet.save_model("test({}).pth".format(round(acc,2)))



