import os
import numpy as np
import matplotlib.pyplot as plt

def get_root_project_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_word_list():
    """
        setiap kata dipisahkan dengan baris pada file list_kata.txt
        return: list
    """
    word_list = []
    with open("./data/list_kata.txt", "r") as input:
        for line in input.readlines():
            word_list.append(line.replace("\n", ""))
            
    word_list = sorted(word_list, key = str.casefold)
    return word_list

def get_video_list(_dir):
    """
        setiap kata dijadikan folder sendiri sendiri pada directory ./video
        return: list
    """
    file_paths = []
    for file in os.listdir(_dir):
        file_paths.append(os.path.join(_dir, file))
    return file_paths

def compute_distance(point_a, point_b):
    """
        compute euclidean distance
        return: float
    """
    return np.sqrt(np.sum((point_b - point_a) ** 2, axis=0))

def compute_distance_adj(point_a, point_b):
    """ if len(point_a)==2:
        point_a = np.append(point_a, [1])
    if len(point_b)==2:
        point_b = np.append(point_b, [1]) """
        
    x1 = point_a[0:2]*point_a[2]
    x2 = point_b[0:2]*point_b[2]
    return np.sqrt(np.sum((x2 - x1) ** 2, axis=0))

def get_mean(*args):
    """
        return: [mean_x, mean_y]
    """
    return np.array([sum([elem[0] for elem in args])/len(args), sum([elem[1] for elem in args])/len(args)])

def get_mean_adj(*args):
    """
        return: [x, y, z]
    """
    return np.array([sum([elem[0]*elem[2] for elem in args])/len(args), sum([elem[1]*elem[2] for elem in args])/len(args)])

def plot(history):
    """
        return: plot acc + loss
    """
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.clf()
