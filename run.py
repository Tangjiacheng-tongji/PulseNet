import numpy as np

from module import *
from meme import MemePools
from tools import Zscore
from model.memenet import MemeNet
from torch.utils.data import TensorDataset, DataLoader

from params import load_ucr_dataset,\
    targets, target_channels, capacity, \
    num_channels, origin_interval, prune_iter, keep_channel
from tools import printr, check_acu, filter

interval = origin_interval

from network import GRUModel

#nets = {"RNN": RNN(1, 64, 1), "LSTM": LSTM(1, 64, 1), "GRU": GRUModel(1, 64, 1)}
net = GRUModel(1, 64, 1)
batch_size = 8


if __name__ == "__main__":
    print("fetching data...")
    data = load_ucr_dataset('../UCRArchive_2018')

    memes = MemePools(capacity=capacity)
    memenet = MemeNet(net, func=printr,device_ids=[])
    rf = memenet.rf

    for name1, used_data1 in data.items():
        train_data1, train_label1, test_data1, test_label1 = used_data1.values()
        res_data, res_label = filter(train_data1, train_label1)

        output_size1 = int(train_label1.max())+1
        s, s2, _ = train_data1.size()

        train_loader1 = DataLoader(TensorDataset(train_data1, train_label1), \
                                  batch_size=batch_size, shuffle=True, num_workers = 2)
        test_loader1 = DataLoader(TensorDataset(test_data1, test_label1), \
                                 batch_size=batch_size)

        memenet.model.fc = nn.Linear(memenet.model.fc.in_features, output_size1)

        '''
        memenet.warm_train(train_loader1, test_loader1, lr=1.5e-3, epochs=5*10**4//s,
                           vis_iter = 200)
        acc = memenet.test_prototype(test_loader1)

        info = "{}-gru:{}\n".format(name1,acc)
        print(info)
        '''

        if s2 <= 100:
            meme_sizes={"rnn":[(10,1)]}
            target_stride = {"rnn": (5, 1)}
        else:
            meme_sizes={"rnn":[(75,1)]}
            target_stride = {"rnn": (10, 1)}

        used_targets, used_channels, used_stride, used_size = memenet.update_name(targets,
                                                                               target_channels, target_stride,
                                                                               meme_sizes)
        zscore = Zscore(used_channels)

        for target in used_targets:
            memenet.add_monitor(target)

        print("start fetching...")

        memenet.fetch_memes(used_channels, used_stride, used_size, num_channels,
                            train_data1, train_label1, zscore, prune_iter, memes,
                            keep_channel, interval)
        kernels = memes.get_kernels()
        masks = memes.get_masks(range(output_size1))

        key_dict, history = memes.retrieve(memenet.optional_node)
        memenet.show_rf(name1, train_data1, key_dict, depository=True, history=history)

        memenet.add_memes(kernels, zscore_ED_mask, zscore.zscore)

        clf = check_acu(memenet, masks, train_data1, train_label1, test_data1, test_label1)
        acc = clf.score(test_data1, test_label1)
        print(acc)
        break

        for name2, used_data2 in data.items():
            if name2 == name1 or name2 in []:
                continue
            print("old task:{}/new task:{}".format(name1, name2))
            train_data2, train_label2, test_data2, test_label2 = used_data2.values()

            train_label2 += output_size1
            test_label2 += output_size1

            train_loader2 = DataLoader(TensorDataset(train_data2, train_label2), \
                                    batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader2 = DataLoader(TensorDataset(test_data2, test_label2), \
                                    batch_size=batch_size)

            output_size2 = int(train_label2.max()) + 1
            s, s2, _ = train_data2.size()

            memenet.model.fc = nn.Linear(memenet.model.fc.in_features, output_size1 + output_size2)


            memenet.warm_train(train_loader2, test_loader2, lr=1.5e-3, epochs=5*10**4 // s,
                                vis_iter=200)

            acc = memenet.test_prototype(test_loader2)

            info = "{}-gru:{}\n".format(name2, acc)
            print(info)


            map_dict, used_labels = memes.remap(memenet, zscore)
            memes = MemePools(capacity=capacity)
            zscore.update(map_dict, True)

            memes.get_memes_from_field(search_dict=map_dict, search_label=used_labels, compare_dict=map_dict,
                                       compare_label=used_labels, target_channels=used_channels,
                                       target_stride=used_stride, num_channels=num_channels,
                                       sizes=used_size, times=interval,
                                       prune_iter=prune_iter, optional_nodes=memenet.optional_node)

            if s2 <= 100:
                for s in used_size.keys():
                    used_size[s] = [(10, 1)]
                for s in used_stride.keys():
                    used_stride[s] = (5, 1)
            else:
                for s in used_size.keys():
                    used_size[s] = [(75, 1)]
                for s in used_stride.keys():
                    used_stride[s] = (10, 1)

            print("start fetching...")


            memenet.fetch_memes(used_channels, used_stride, used_size, num_channels,
                                train_data2, train_label2, zscore, prune_iter, memes,
                                keep_channel, interval)

            kernels = memes.get_kernels()

            memenet.add_memes(kernels, zscore_ED_mask, zscore.zscore)

            all_data = [test_data1,test_data2]
            all_label = [test_label1,test_label2]

            mean_acu, std_acu = check_acu(memenet, train_data2, train_label2, test_data1, test_label1)
            print("old data:",mean_acu, std_acu)

            mean_acu, std_acu = check_acu(memenet, train_data2, train_label2, test_data2, test_label2)
            print("new data:",mean_acu, std_acu)

            mean_acu, std_acu = check_acu(memenet, [train_data1, train_data2], [train_label1, train_label2], test_data1, test_label1)
            print("all data/old:",mean_acu, std_acu)

            mean_acu, std_acu = check_acu(memenet, [train_data1, train_data2], [train_label1, train_label2], test_data2, test_label2)
            print("all data/new:", mean_acu, std_acu)

            mean_acu, std_acu = check_acu(memenet, [res_data,train_data2], [res_label,train_label2], test_data1, test_label1)
            print("combined data/old:",mean_acu, std_acu)

            mean_acu, std_acu = check_acu(memenet, [res_data,train_data2], [res_label,train_label2], test_data2, test_label2)
            print("combined data/new:",mean_acu, std_acu)
            memenet.del_memes()
            break
        break
    del memenet, memes
