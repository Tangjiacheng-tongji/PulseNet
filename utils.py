import numpy as np
import cupy as cp
from tools import transform
from numba import njit

def compute_ed(shapelet1,shapelet2):
    #shapelet1:N1*channels*l*w
    #shapelet2:channels*l*w
    #output:N1
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(1,2,3)))
    return norm

def ed_channel1(shapelet1,shapelet2):
    #shapelet1:N1*channels*l*w
    #shapelet2:M*channels*l*w
    #diff:N1*M*channels*l*w
    #norm:N1*M
    #norm2:N1*M*channels
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    shapelet1=np.expand_dims(shapelet1,1).repeat(len(shapelet2),1)
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(3,4)))
    return norm
    
def ed_channel2(shapelet1,shapelet2):
    #shapelet1:N*K*channels*l*w
    #shapelet2:M*channels*l*w
    #diff:N*K*M*channels*l*w
    #norm:N*K*M
    #norm2:N*K*M*channels
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    shapelet1=np.expand_dims(shapelet1,2).repeat(len(shapelet2),2)
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(4,5)))
    return norm

def shapelet2maps(fields,shapelets):
    #shapelets:a collection of shapelets(M*channels*l*w)
    #receptive field:list of N feature with all extracted receptive field
    #list of N lists, where a collection of shapelets is included
    #output:(array)N*M*channels
    if fields.ndim!=1:
        dist=np.min(ed_channel2(fields,shapelets),axis=1)
    else:
        dist=np.array([np.min(ed_channel1(i,shapelets),axis=0) for i in fields])
    return dist

def compute_percentile(candidates,n,k):
    #Set a threshold to distinguish similar memes
    shapelet1=candidates[np.random.choice(len(candidates),n,replace=True)]
    shapelet2=candidates[np.random.choice(len(candidates),n,replace=True)]
    dist=compute_ed(shapelet1,shapelet2)
    del shapelet1
    del shapelet2
    return np.percentile(dist,k)

def compute_percentile_channels(candidates,n,k,channels):
    #Set thresholds for specified channels
    threshold_dict=dict()
    shapelet1=candidates[np.random.choice(len(candidates),n,replace=True)]
    shapelet2=candidates[np.random.choice(len(candidates),n,replace=True)]
    dist = np.sqrt(np.sum((shapelet1-shapelet2)**2,axis=(-1, -2))).T
    for choice in channels:
        threshold_dict[transform(choice)]=np.percentile(dist[choice], k)
    del shapelet1
    del shapelet2
    return threshold_dict

def compute_terms(A, B, batch_size = 1000):
    ans = []
    for i in range(0,len(A),batch_size):
        used_A = A[i:i+batch_size]
        length = B.shape[1]
        px = (cp.sum(used_A, axis = -1, keepdims = True) / length)
        py = (cp.sum(B, axis = -1, keepdims = True) / length).T
        pxy = (cp.matmul(used_A,B.T) / length)
        result = cp.multiply(pxy,cp.log2(cp.true_divide(cp.add(pxy,1e-8),cp.add(cp.matmul(px,py),1e-8))))
        ans.append(cp.where((px==0)*(py==0),2,
                cp.where((px==0)+(py==0)+(pxy==0),1,result)))
    return cp.concatenate(ans)

def cal_condition_ents(A, B, update_batch_size = 1000):
    # compute mutual information
    mask = cp.hstack((cp.ones((len(A), len(B))),
                      cp.tril(np.ones((len(A), len(A)))) - cp.eye(len(A))))
    A = cp.asarray(A)
    B = cp.vstack((cp.asarray(B), A))
    result = cp.add(cp.add(cp.add(compute_terms(A, B, update_batch_size),compute_terms(1 - A, B, update_batch_size)),
            compute_terms(A, 1 - B, update_batch_size)), compute_terms(1 - A, 1 - B, update_batch_size))
    ans = cp.multiply(result, mask)
    weight = cp.arange(len(mask))
    weight[0] = 1
    return cp.asnumpy(np.sum(ans, axis=1)/weight)

def cal_ent(used_p):
    result = np.zeros_like(used_p)
    index = (used_p != 0)&(used_p != 1)
    hold = used_p[index]
    result[index] = -hold * np.log2(hold)- (1 - hold) * np.log2(1 - hold)
    return result

def get_ans(target, size):
    length = target.shape[1] + 1
    pl = np.arange(1,length)/(target + 1)
    pr = (length - np.arange(1,length))/(size - target - 1)
    rate = (target + 1)/size
    ans = (cal_ent(pl) * rate + (1 - rate) * cal_ent(pr)).reshape(-1)
    target = target.reshape(-1)
    min_idx = np.argmin(ans)
    return ans[min_idx], target[min_idx]

def get_ans_spec(data, size):
    if len(data) == 0:
        return 1,0
    else:
        return 0,data[0]

def compute_minent(A):
    length, size = A.shape
    min_entropy = np.zeros(length)
    target = np.zeros(length, dtype = np.int16)
    for i in range(length):
        data = A[i].nonzero()[0]
        if len(data) <= 1:
            ig,idx = get_ans_spec(data, size)
        else:
            ig,idx = get_ans(np.vstack((data[1:] - 1, data[:-1])), size)
        min_entropy[i] = ig
        target[i] = idx
        #break
    return min_entropy, target

@njit
def get_entropy(used_label):
    init_ent=np.zeros((np.max(used_label)+1))
    for i in np.unique(used_label):
        init_ent[i]=compute_entropy(np.where(used_label==i,1,0),len(used_label))
    return init_ent

@njit
def compute_entropy(data,length):
    assert data.ndim==1
    if length==0:
        return 0
    else:
        if data.ndim==1:
            p = np.sum(data) / length
            if p==0 or p==1:
                return 0
            else:
                return -(p*np.log2(p)+(1-p)*np.log2(1-p))

@njit
def get_threshold(datas,idx,target):
    threshold=[]
    for i in range(len(datas)):
        data=datas[i]
        index=int(target[i])
        seq=idx[i]
        threshold.append((data[seq[index]]+data[seq[index+1]])/2)
    return threshold

@njit
def cal_entropy(label):
    p=np.sum(label)/len(label)
    if p==0 or p==1:
        return 0
    else:
        return -(p*np.log2(p)+(1-p)*np.log2(1-p))

@njit
def compute_term(A,B):
    total = len(A)
    px = np.sum(A) / total
    py = np.sum(B) / total
    pxy = np.sum(A * B) / total
    if px == 0 and py == 0:
        return 1
    elif px == 0 or py == 0 or pxy == 0:
        return 0
    return pxy * np.log2(pxy / (px * py))

@njit
def cal_condition_ent(A,B):
    #compute mutual information
    return compute_term(A, B) + compute_term(1 - A, B) + \
           compute_term(A, 1 - B) + compute_term(1 - A, 1 - B)

@njit
def get_corr(i, meme_ans, dict_ans):
    item = meme_ans[i]
    ent = cal_entropy(item)
    sum_info = 0
    list_length = len(dict_ans)
    for j in range(i):
        sum_info += cal_condition_ent(item, meme_ans[j])
    for j in range(len(dict_ans)):
        sum_info += cal_condition_ent(item, dict_ans[j])
    if list_length + i != 0:
        return sum_info / ((list_length + i) * (ent+1e-6))
    else:
        return 0

    '''
    #The memes with lower entropy are excluded directly
    item = meme_ans[i]
    ent = cal_entropy(item)
    if ent ==0:
        return 999
    else:
        sum_info = 0
        list_length = len(meme_ans)
        for j in range(i):
            sum_info += cal_condition_ent(item, meme_ans[j])
        if list_length != length:
            for k in range(length,list_length):
                sum_info += cal_condition_ent(item, meme_ans[k])
        if list_length - length + i != 0:
            return ent - sum_info / (list_length - length + i)
        else:
            return 0
    '''