import cv2
import numpy as np
import os
import random
import shutil

"""
本文件用于对数据集按比例随机划分为训练集和测试集
"""

class SplitDataset:

    # rate为训练集的比例
    def __init__(self, rate, orignImgPath, orignLabelPath):
        self.rate = rate
        self.orignImgPath = orignImgPath
        self.orignLabelPath = orignLabelPath

    # trainPath存放训练集，testPath存放测试集
    def split(self, trainPath, testPath):
        pathes = [trainPath + "/images", trainPath + "/labels", testPath + "/images", testPath + "/labels"]
        for path in pathes:
            if not os.path.exists(path):
                os.makedirs(path)

        filenames = os.listdir(self.orignImgPath)
        random.shuffle(filenames)
        num = len(filenames)
        i = 0
        for filename in filenames:
            print(filename)
            # 训练集
            if i < self.rate * num :
                shutil.copy(self.orignImgPath + "/" + filename, trainPath + "/images/" + filename)
                shutil.copy(self.orignLabelPath + "/" + filename[:-4] + ".png",
                            trainPath + "/labels/" + filename[:-4] + ".png")
            # 测试集
            else:
                shutil.copy(self.orignImgPath + "/" + filename, testPath + "/images/" + filename)
                shutil.copy(self.orignLabelPath + "/" + filename[:-4] + ".png",
                            testPath + "/labels/" + filename[:-4] + ".png")

            i += 1


if __name__=="__main__":
    processor = SplitDataset(0.8, "swinyseg/images", "swinyseg/GTmaps")
    processor.split("swinyseg/train", "swinyseg/test")

