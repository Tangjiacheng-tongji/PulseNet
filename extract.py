import numpy as np

def generate_candidates(data,labels,channels,
                         size=(1,1),stride=(1,1), indice = []):
    #get memes for data
    #size:fixed size of memes
    #stride:extraction stride
    candidates = []
    infos = []
    target_channel = []
    for i in range(len(data)):
        feature_maps, label, channel = data[i], labels[i], channels[i]
        for l in range(0,feature_maps.shape[1]-size[0]+1,stride[0]):
            for w in range(0,feature_maps.shape[2]-size[1]+1,stride[1]):
                candidates.append(feature_maps[channel,l:l+size[0],w:w+size[1]])
                if len(indice) == 0:
                    infos.append([label, i, l, w])
                else:
                    infos.append([label, indice[i], l, w])
                target_channel.append(channel)
    return np.array(candidates),np.array(infos),np.array(target_channel)

def generate_candidates_with_position(data,labels,positions,size=(1,1)):
    assert len(data)==len(labels)
    candidates=[]
    infos=[]
    for i in range(len(data)):
        feature_maps, label = data[i], labels[i]
        if feature_maps.shape[1]!=size[0] or feature_maps.shape[2]!=size[1]:
            continue
        else:
            candidate=[]
            info=[]
            for l in range(feature_maps.shape[1]-size[0]+1):
                for w in range(feature_maps.shape[2]-size[1]+1):
                    candidate.append(feature_maps[:,l:l+size[0],w:w+size[1]])
                    idx,x0,y0,_,_=positions[i]
                    info.append((label,(idx,x0+l,y0+w)))
            candidates.append(candidate)
            infos.append(info)
    return np.array(candidateats),infos

def fast_generate(data,size=(1,1)):
    candidates=[]
    for l in range(data.shape[1]-size[0]+1):
        for w in range(data.shape[2]-size[1]+1):
            candidates.append(data[:,l:l+size[0],w:w+size[1]])
    return np.array(candidates)
