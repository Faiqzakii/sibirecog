import os
import numpy as np


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
        Cartesian Distance
    """
    return np.sqrt(np.sum((point_a - point_b) ** 2, axis=0))

def get_mean(*args):
    """
        [mean_x, mean_y]
    """
    return np.array([sum([elem[0] for elem in args])/len(args), sum([elem[1] for elem in args])/len(args)])
