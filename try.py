from params import load_ucr_dataset
import matplotlib.pyplot as plt

data = load_ucr_dataset('../UCRArchive_2018')

for name1, used_data1 in data.items():
    train_data1, train_label1, test_data1, test_label1 = used_data1.values()
    y = train_data1[1][:,0]
    x = list(range(len(train_data1[1])))
    plt.plot(x,y)
    plt.show()
    break
