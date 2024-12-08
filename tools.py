import json
import os
import copy
import torch
from module import Whitening, memes_ED_mask
import numpy as np
import random
from itertools import permutations

class MemeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, range):
            return list(obj)
        elif isinstance(obj,np.int64) or isinstance(obj,np.int32):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif torch.is_tensor(obj):
            return obj.numpy().tolist()
        else:
            return json.JSONEncoder.default(self, obj)

class Zscore:
    def __init__(self,target_channels):
        self.zscore = dict()
        for target,channels in target_channels.items():
            #self.zscore[target] = nn.BatchNorm2d(len(channels), affine=False)
            self.zscore[target] = Whitening(channels)
    def update(self, map_dict, action = True):
        for key,value in map_dict.items():
            if action:
                map_dict[key]= self.zscore[key](value).cpu().detach().numpy()
            else:
                map_dict[key] = value.detach().numpy()

    def save(self, filename):
        with open(filename, 'w') as f:
            for target in self.zscore:
                data = dict()
                zscore_module = self.zscore[target]
                data["mean"] = zscore_module.mean.detach().numpy()
                data["var"] = zscore_module.var.detach().numpy()
                data["length"] = zscore_module.length
                data["eps"] = zscore_module.eps
                data["channels"] = zscore_module.channels
                data["target"] = target
                json.dump(data, f, cls=MemeEncoder)
                f.write('\n')
        print("Zscore data saved as " + filename)
    def load(self,filename):
        print("Load Zscore data from " + filename)
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                target = data["target"]
                if target not in self.zscore:
                    if "module." + target in self.zscore:
                        target = "module." + target
                    elif target[7:] in self.zscore:
                        target = target[7:]
                self.zscore[target].load(data["mean"], data["var"],
                                    data["length"], data["eps"])
    def clear(self):
        for target in self.zscore:
            self.zscore[target] = Whitening(len(self.zscore[target].mean))
    def end_update(self):
        for whitening in self.zscore.values():
            whitening.switch()

    def cuda(self):
        for whitening in self.zscore.values():
            whitening.cuda()

    def cpu(self):
        for whitening in self.zscore.values():
            whitening.cpu()

def printr(msg,log="log.txt",rec=False):
    if rec:
        print(msg)
        with open(log,'a') as logfile:
            print(msg,file=logfile)
    else:
        print(msg)

def fusion(memes1,memes2,real_idx1,real_idx2):
    #fuse 2 memes to 1 meme
    #real_idx1 means the real idx of the pictures(used to analysis combined data)
    temp1=copy.deepcopy(memes1)
    temp2=copy.deepcopy(memes2)
    temp1.update_indices(real_idx1)
    temp2.update_indices(real_idx2)
    capacity=max(temp1.capacity,temp2.capacity)
    temp1.update_pools([],capacity)
    for pools in temp2.meme_dict.values():
        for pool in pools.values():
            for meme2 in pool:
                temp1.update_meme(meme2)
    return temp1

def get_latest_checkpoint(save_root):
    files = os.listdir(save_root)
    assert len(files)!=0
    files.sort(key=lambda x:os.path.getmtime(save_root +'/'+x))
    return files[-1]

def touch(log):
    if not os.path.exists(log):
        open(log,'a').close()

def transform(channel):
    #input(list or numpy.ndarray):index of channel
    #output(str):visualize the index
    if isinstance(channel,list) or isinstance(channel,np.ndarray):
        return ','.join(map(lambda x:str(x),channel))
    else:
        return str(channel)

def cal_norm(meme):
    norm=np.maximum(np.sqrt(np.sum(meme**2,axis=(-1,-2,-3),keepdims=True)),1e-6)
    normalized_meme=meme/norm
    return normalized_meme,np.squeeze(norm)

def compute_similarity(memes1,memes2):
    #meme1:array like
    #expression of meme1
    #meme2:array like
    #expression of meme2
    assert memes1.shape==memes2.shape
    target=np.array(list(permutations(range(memes1.capacity))))
    #get all possible permutation
    per_memes2=memes2[target]
    normalized_meme1,norm1=cal_norm(memes1)
    normalized_meme2,norm2=cal_norm(per_memes2)
    weight=1-np.abs(norm1-norm2)/np.maximum(norm1,norm2)
    #inner_product=np.sum(normalized_meme1*normalized_meme2,axis=(-1,-2,-3))
    diff=(normalized_meme1-normalized_meme2)
    inner_product=1-np.sqrt(np.sum(diff**2,axis=(-1,-2,-3)))/2
    #use this to avoid all 0 array
    simliarity=np.max(np.mean(weight*inner_product,axis=1))
    return simliarity

def compare(memes1,memes2):
    similar_dict=dict()
    meme_dict1=memes1.meme_dict
    meme_dict2=memes2.meme_dict
    for label1,pools1 in meme_dict1.items():
        similar_dict[label1]=dict()
        for channel1,pool1 in pools1.items():
            if label1 in meme_dict2 and channel1 in meme_dict2[label1]:
                meme1=np.array([item.meme for item in pool1])[:,pool1[0].channel]
                meme2=np.array([item.meme for item in meme_dict2[label1][channel1]])[:,pool1[0].channel]
                similar_dict[label1][channel1]=compute_similarity(meme1,meme2)
            else:
                similar_dict[label1][channel1]=0
    for label2,pools2 in meme_dict2.items():
        if label2 not in meme_dict1:
            similar_dict[label2]=None
        else:
            for channel2 in pools2:
                if channel2 not in meme_dict1[label2]:
                    similar_dict[label2][channel2]=0
    return similar_dict

def update_similarity(similarity_dict,choice):
    similarity=copy.deepcopy(similarity_dict)
    assert similarity.keys()==choice.keys()
    for label,sim in similarity.items():
        channels=list(sim.keys())
        for channel in channels:
            if channel not in choice[label]:
                sim.pop(channel)
    return similarity

def get_regions(data,positions):
    #Get representation in a specific location
    regions = []
    for position in positions:
        id, x0, y0, x1, y1 = position
        region = data[id][:, x0:x1, y0:y1]
        regions.append(region)
    return regions

def get_typical_img(positions,return_index=False):
    return np.unique(np.array([position[0] for position in positions]),return_index=return_index)

def sparse_sampling(data1,label1,data2,label2,positions,times=4):
    used_id = get_typical_img(positions)
    sample_id = np.random.choice(len(data2), len(used_id) * times)
    train_data = np.concatenate([data1[used_id], data2[sample_id]])
    train_label = np.concatenate([label1[used_id], label2[sample_id]])
    return train_data,train_label

def update_info(infos,idx):
    new_infos=[]
    for info in infos:
        new_info=[]
        for item in info:
            label,indice=item
            new_info.append([label,[idx[indice[0]],indice[1],indice[2]]])
        new_infos.append(new_info)
    return new_infos

def get_images(data_loader,positions):
    '''todo: update saving pattern'''
    images=[]
    idx=get_typical_img(positions)
    batch_size=data_loader.batch_size
    id=0
    for i, (image, _) in enumerate(data_loader):
        for s in range(id,len(idx)):
            if int(idx[s]/batch_size)<i:
                continue
            elif int(idx[s]/batch_size)==i:
                images.append(image[(idx[s])%batch_size].detach().numpy())
            else:
                break
        id=s
        if len(idx)-1==id:
            break
    idx_dict = dict()
    for i in range(len(idx)):
        idx_dict[idx[i]] = i
    new_positions=[]
    for position in positions:
        new_positions.append([idx_dict[position[0]],position[1],position[2],position[3],position[4]])
    idx = dict((v, k) for k, v in idx_dict.items())
    return torch.Tensor(images), new_positions, idx

def memes2kernels(memes):
    kernels = dict()
    for count in range(len(memes)):
        item = memes[count]
        size = (item.meme.shape[0],item.meme.shape[1])
        where = item.where
        key = "{}_{}_{}".format(where, size[0], size[1])
        if key not in kernels:
            kernels[key] = {"kernel": [], "thresholds": [], "index": []}
        kernels[key]["kernel"].append(item.meme)
        kernels[key]["thresholds"].append(np.mean(item.threshold))
        kernels[key]["index"].append([count,item.channel])
    return kernels

def threshold_classify(kernels, field_dict):
    #kernels: dict like result of memes.get_kernel
    from model.tools import compute_min_distance
    threshold = []
    dist = []
    idx = []
    for key in kernels:
        memes = kernels[key]
        where = int(key.split("_")[0])
        feature_map = field_dict[where]

        meme_kernel = np.array(memes["kernel"], dtype=np.float32)
        meme_channel = [item[1] for item in memes["index"]]
        min_dist = compute_min_distance(meme_kernel, meme_channel, feature_map)
        dist.append(min_dist)
        threshold.append(memes["thresholds"])
        idx += memes["index"]
    dist = np.concatenate(dist, axis=1)
    threshold = np.expand_dims(np.concatenate(threshold), axis=0)
    return (dist < threshold).T,idx

def get_layers(kernels):
    layers=dict()
    for where in kernels:
        layers[where] = dict()
        for label in kernels[where]:
            layers[where][label] = []
            meme_pool = kernels[where][label]
            for size,memes in meme_pool.items():
                meme_kernel=torch.FloatTensor(memes["kernel"])
                meme_channel=torch.Tensor(memes["channel"]).long()
                layers[where][label].append(memes_ED_mask(meme_kernel, meme_channel))
    return layers

def read_rec_txt(file_path):
    dataset_model_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines)//3*3, 3):
            dataset_model = lines[i].split(":")[0]
            dataset, model = dataset_model.split('-')
            if dataset not in dataset_model_dict:
                dataset_model_dict[dataset] = []
            dataset_model_dict[dataset].append(model)
    return dataset_model_dict

def check_acu(memenet, mask, train_data1, train_label1, test_data1, test_label1, batch_size = 64, epoch=3):
    from sklearn import svm
    if isinstance(train_data1, list):
        recs = []
        for data in train_data1:
            recs += [memenet.get_features(data[i:i + batch_size]).cpu().numpy() \
                    for i in range(0, len(data), batch_size)]
        rec = np.vstack(recs)
        lab = torch.cat(train_label1).numpy()
    else:
        rec = np.vstack(
            [memenet.get_features(train_data1[i:i + batch_size]).cpu().numpy() \
             for i in range(0, len(train_data1), batch_size)])
        lab = train_label1.numpy()
    acu = []
    for i in range(epoch):
        clf = svm.SVC(kernel='linear', max_iter=2 * 10 ** 5)
        clf.fit(rec[i:i - 3], lab[i:i - 3])
        if isinstance(test_data1, list):
            cnt = 0
            for data, label in zip(test_data1, test_label1):
                s2 = len(data)
                rec2 = np.vstack(
                    [memenet.get_features(data[i:i + batch_size]).cpu().numpy() for i in
                     range(0, s2, batch_size)])
                lab2 = label.numpy()
                acc = clf.score(rec2, lab2) * s2
                cnt += s2
            acu.append(acc/cnt)
        else:
            s2 = len(test_data1)
            rec2 = np.vstack(
                [memenet.get_features(test_data1[i:i + batch_size]).cpu().numpy() for i in range(0, s2, batch_size)])
            lab2 = test_label1.numpy()
            acc = clf.score(rec2, lab2)
            acu.append(acc)
    acu = np.array(acu)
    print(np.mean(acu), np.std(acu))
    return clf

def filter(data, label, percentage = 0.3):
    info = dict()
    for dat, lab in zip(data, label):
        s = int(lab)
        if s not in info:
            info[s] = list()
        info[s].append(dat.numpy())
    output_data = []
    output_label = []
    for key, values in info.items():
        s = max(int(len(values) * percentage),1)
        index = [i for i in range(len(values))]
        np.random.shuffle(index)
        output_data.append(np.array(values)[index][:s])
        output_label += [key] * s
    return torch.tensor(np.concatenate(output_data)), torch.tensor(output_label)