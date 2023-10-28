import matplotlib.pyplot as plt
import numpy as np
import statistics


def plot_distribution(np_data, name_save = "", save = True ,show = False):
    '''

    :param np_data: numpy data to plot distribution visulization
    :param save: command to save fig plot
    :param name_save: name and path to save  fig plot
    :param show: command to show
    :return:
    '''
    # print("here")
    sd = statistics.stdev(np_data)
    fig, ax = plt.subplots()
    plt.hist(np_data, bins=100)
    plt.gca().set(title='Frequency Histogram'+str(sd), ylabel='Frequency')
    if show:
        plt.show()
    if save:
        if name_save[-4:-1] in [".jpg", ".bmp"]:
            name_save = name_save[0:-4]
        else:
            name_save = name_save
        fig.savefig(name_save+"_dis"+".png")

def plot_data(np_data, name_save = "", save = True,show = False):
    '''

    :param np_data: numpy data to plot data visulization
    :param save: command to save fig plot
    :param name_save: name and path to save  fig plot
    :param show: command to show
    :return:
    '''
    mean = statistics.mean(np_data)
    fig, ax = plt.subplots()
    plt.plot(np_data)
    plt.ylabel('some numbers')
    plt.xlabel('order'+str(mean))
    if show:
        plt.show()
    if save:
        if name_save[-4:-1] in [".jpg",".bmp"]:
            name_save = name_save[0:-4]
        else:
            name_save = name_save
        fig.savefig(name_save+"_data"+".png")

def average_in_sd(data, sd_lower = 1, sd_upper = 1):
    mean = statistics.mean(data)
    sd = statistics.mean(data)
    new_data = []
    for each_data in data:
        if each_data > mean-sd_lower*sd and each_data< mean+sd_upper*sd:
            new_data.append(each_data)
    # print("new_data",new_data)
    if new_data == []:
        new_data = data
    new_mean = statistics.mean(new_data)
    return new_mean

if __name__ == '__main__':
    data = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 14, 15, 1, 6, 2, 2, 4, 4, 6, 6, 6, 6]
    np_data = np.array(data)
    # print(data.shape)
    x = np.random.normal(size=100)
    plot_data(data,save=True,name_save="123")
