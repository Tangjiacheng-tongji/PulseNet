import utils
import numpy as np
import multiprocessing
from model.tools import compute_min_distance

def get_sample(target,func=np.square):
    #target:array_like
    #channels*1*batch_number*l*w
    target_sum = np.sum(target, axis=(-2, -1))
    return get_idx(target_sum, func)

def get_idx(target_sum,func=np.square):
    #Calculate sum of feature representation
    mean = np.mean(target_sum,axis=0,keepdims=True)
    dist = np.array([func(diff) for diff in (target_sum-mean)])
    target_idx=np.argmin(dist,axis=0)
    return target_idx

def get_target(target, meme_dict, times = 0.5):
    import torch
    result = []
    for meme in meme_dict:
        dist = meme(torch.Tensor(target)).detach()
        min_dist = torch.min(torch.min(dist,3).values, 2).values.numpy()
        std = times * np.std(min_dist, axis=0, keepdims=True)
        ans = min_dist > std
        result.append(np.max(ans, axis=1))
    return np.min(np.array(result), axis = 0)

def get_intersect(result,ans):
    intersection = []
    for item in result:
        choice = [s for s in item if ans[s]]
        intersection.append(choice)
    return intersection

def get_result(feature_map,labels,candidates,used_info,channels,init_ent):
    #get result in a absolutely fast path
    #useful when there are lots of memes
    assert len(candidates)==len(channels)
    dist = compute_min_distance(candidates, channels, feature_map)
    assert dist.ndim==2
    assert dist.shape[1]==len(used_info)

    idx = np.argsort(dist, axis=0)
    candidate_label = used_info[:,0]
    direction_info = used_info[:,1:]
    real_idx = np.array(
        [np.where(labels[idx][:, i] == candidate_label[i], 1, 0) for i in range(len(candidate_label))])
    min_entropy, target = utils.compute_minent(real_idx)
    ig=init_ent[candidate_label]-min_entropy
    thresholds = utils.get_threshold(dist.T,idx.T,target)
    return candidates, candidate_label, ig, thresholds, \
           direction_info, channels

def get_candidates(used_field,used_data,used_label,used_info,label,concerned_channel):
    #used_field:batch_size*num_for_target*channel*l*w
    #available batches of memes,find out the most useful batch
    #used_data:batch_size*channel*l*w
    #can be decomposed to used_fields,used to get mean data
    #get important channel from channels
    #target:number of channel(s)*(flexible)channel*l*w
    #calculate the sum of spatial infomation
    #target_idx:find the closest idx
    #concerned_channel:array like
    #suitable for simple channel,compound channels is to be done
    assert len(used_field) == len(used_data)
    assert len(used_field) == len(used_label)
    idx=np.where(used_label==label)
    target=np.array(used_data[idx].transpose(1,0,2,3)[concerned_channel])

    target_idx=idx[0][get_sample(target)]
    assert len(target_idx)==len(concerned_channel)
    output_channel=np.repeat(concerned_channel,[len(used_field[i]) for i in target_idx],axis=0)
    return np.concatenate(used_field[target_idx]),np.concatenate(np.array(used_info)[target_idx]),output_channel

def get_candidates_in_class(used_data, used_label,
                            concerned_channel, target_label,
                            times = 1.5, meme_dict = {}):
    results = []
    channels = []
    for label in target_label:
        use_dict = meme_dict[label] if label in meme_dict else []

        idx = np.where(used_label == label)

        result, length = get_candidates_with_sample(used_data[idx],
                                    concerned_channel, times, use_dict)

        target_idx = idx[0][result]
        results.append(target_idx)
        channels.append(np.repeat(concerned_channel, length, axis=0))
    return np.concatenate(results), np.concatenate(channels)

def get_candidates_with_sample(target, concerned_channel,
                               times = 1.5, use_dict = {}):
    #latest version of function get_candidates
    #more objects are selected

    concerned_target = target[:, concerned_channel, :, :]
    result = sample_from_subclasses(concerned_target, times)
    if len(use_dict):
        # Select outliers whose distance exceeds 1.5 times the standard deviation of the existing memes
        # As additional sampling points
        ans = get_target(target, use_dict, times=1.5)
        result = get_intersect(result, ans)

    return np.concatenate(result).astype(np.int64), \
           [len(item) for item in result]

def get_points(target, times = 1.5):
    sample_idx = []
    target_idx = get_sample(target, func=np.square)
    sample_idx.append(target_idx)
    std = target[target_idx]
    dist = np.sum((target - std)**2, axis=(-2,-1))
    dist_idx = np.argsort(dist)
    sort_dist = dist[dist_idx]

    deviation = np.diff(sort_dist)
    thres = times * np.std(deviation)

    check_point = np.where(deviation > thres)[0]
    for i in range(len(check_point)):
        if i == len(check_point) - 1:
            new_idx = dist_idx[check_point[i]+1:]
        else:
            new_idx = dist_idx[check_point[i]+1:check_point[i+1]+1]
        target_idx = get_sample(target[new_idx], func=np.square)
        sample_idx.append(new_idx[target_idx])
    return sample_idx

def get_points_iter(target, target_idx, sample_idx, times = 1.5):
    if len(target) == 1:
        sample_idx.append(target_idx[0])
    else:
        std_idx = get_sample(target, func=np.square)
        sample_idx.append(target_idx[std_idx])

        dist = np.sum((target - target[std_idx])**2, axis=(-2,-1))
        dist_idx = np.argsort(dist)
        deviation = np.diff(dist[dist_idx])
        thres = times * np.std(deviation)
        checkpoint = deviation > thres
        if checkpoint.any():
            used_idx = dist_idx[(np.argmax(checkpoint) + 1):]
            get_points_iter(target[used_idx], target_idx[used_idx], sample_idx,
                            times = times)

def sample_with_kmeans(data, n = 4, func = np.abs, times = 0.1):
    data_sum = np.sum(data,axis=(-2,-1))
    data_mean = np.mean(data_sum)
    idx = np.argsort(func(data_sum-data_mean))
    length = int(len(idx)/n + 0.5)
    groups = [data_sum[idx[i*length:(i+1)*length]] for i in range(n)]
    group_mean = np.array([np.mean(group) for group in groups])
    while True:
        allocation = np.argmin(func(np.expand_dims(data_sum,1)-np.expand_dims(group_mean,0)),axis = 1)
        groups = [data_sum[allocation==i] for i in range(len(groups)) if (allocation==i).any()]
        new_mean = np.array([np.mean(group) for group in groups])
        if len(group_mean) == len(new_mean) and (group_mean == new_mean).all():
            break
        else:
            group_mean = new_mean
    sample_idx=[]
    for i in range(len(groups)):
        group = groups[i]
        idx = np.argsort(func(group-np.mean(group)))
        sample_dict = [idx[0]]
        for item in np.where(np.random.rand(min(len(group),4)) <= times)[0]:
            if item == 0:
                sample_dict.append(idx[-1])
            elif item == 1:
                sample_dict.append(idx[len(idx)//2])
            elif item == 2:
                sample_dict.append(idx[-2])
            else:
                sample_dict.append(idx[len(idx)//2+1])
        sample_idx.append(np.where(allocation==i)[0][list(set(sample_dict))])
    return np.concatenate(sample_idx)

def sample_from_subclasses(targets, times = 1.5):
    #targets: num*channel*length*width
    if len(targets) == 1:
        return np.zeros((targets.shape[1],targets.shape[0])).astype(np.int32)
    else:
        sample_idxes = []
        for i in range(targets.shape[1]):
            target = targets[:,i,:,:]

            '''
            sample_idx = []
            get_points_iter(target, np.arange(len(target)),
                            sample_idx = sample_idx, times = times)
            sample_idxes.append(sample_idx)
            '''
            sample_idxes.append(get_points(target, times))
            #sample_idxes.append(sample_with_kmeans(target))
        return sample_idxes


def compute_result(search_field, search_data, search_label, search_info,
                   compare_map, compare_label, channels, label, init_ent, times = 1.5):
    candidates, info, used_channel = get_candidates_in_class(search_field, search_data, search_label,
                                                                     search_info, channels, label, times)
    result = get_result(compare_map, compare_label, candidates, info, used_channel, init_ent)
    return result

def get_results(search_field, search_data, search_label, search_info,
                compare_map, compare_label, target_channels, num_channel, num_label, times = 1.5):
    #采用该策略并行计算只需20min左右即可完成模因提取工作
    pool = multiprocessing.Pool(processes=8)
    results = []
    available_label = np.unique(compare_label)
    init_ent = utils.get_entropy(compare_label)
    for j in range(0, len(target_channels), num_channel):
        channels = target_channels[j:min(j + num_channel, len(target_channels))]
        for idx in range(0, len(available_label), num_label):
            label = available_label[idx:min(idx + num_label, len(available_label))]
            results.append(pool.apply_async(compute_result, args=[search_field, search_data, search_label, search_info,
                                    compare_map, compare_label, channels, label, init_ent, times]))
    pool.close()
    pool.join()
    return results

def recount(meme_kernel, meme_channel, feature_map, labels, label, init_ent):
    dist = compute_min_distance(meme_kernel, meme_channel, feature_map)
    idx = np.argsort(dist, axis=0)
    real_idx = np.array(
        [np.where(labels[idx][:, i] == label[i], 1, 0) for i in range(len(label))])
    min_entropy, target = utils.compute_minent(real_idx)
    ig = init_ent[label] - min_entropy
    thresholds = utils.get_threshold(dist.T, idx.T, target)
    return ig, thresholds