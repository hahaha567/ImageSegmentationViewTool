import cv2
import numpy as np
import os

"""
本文件用于对VOC2012数据集的预测结果进行可视化以及利用类激活图生成热图
"""
CLASS2COLOR = {1:(0, 0, 128), 2:(0, 128, 0), 3:(0, 128, 128), 4:(128, 0, 0),
        5:(128, 0, 128), 6:(128, 128, 0), 7:(128, 128, 128), 8:(0, 0, 64),
        9:(0, 0, 192), 10:(0, 128, 64), 11:(0, 128, 192), 12:(128, 0, 64),
        13:(128, 0, 192), 14:(128, 128, 64), 15:(128, 128, 192), 16:(0, 64, 0),
        17:(0, 64, 128), 18:(0, 192, 0), 19:(0, 192, 128), 20:(128,64,0)}

class ViewVOC2012:
    def __init__(self):
        pass

    # 对imgpath文件夹下的预测结果进行可视化处理, 并保存至savePath
    def process(self, imgPath, savePath):
        filenames = os.listdir(imgPath)
        for filename in filenames:
            # 过滤文件
            #if not filename.endswith(".png"):
            #    continue
            path = imgPath + "/" + filename
            outPath = savePath + "/" + filename
            out = self.view(path)
            cv2.imwrite(outPath, out)

    def view(self, imgName):
        img = cv2.imread(imgName, -1)
        result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int)
        for i in range(1,21):
            result[img == i] = CLASS2COLOR[i]
        return result

    # 对图片尺寸进行修改
    # imgPath为存放原始数据文件夹 savePath为存放输出数据文件夹 height为目标高度  weight为目标宽度
    def reshape(self, imgPath, savePath, height, weight):
        filenames = os.listdir(imgPath)
        for filename in filenames:
            path = imgPath + "/" + filename
            outPath = savePath + "/" + filename
            img = cv2.imread(path, -1)
            img = cv2.resize(img, (weight, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(outPath, img)


    # imgPath文件夹下存放原图，activateMapPath文件夹下存放类激活图，savePath为存放生成的热图的目录
    def getHeatMap(self, imgPath, activateMapPath, savePath):
        filenames = os.listdir(activateMapPath)
        for filename in filenames:
            # 过滤文件
            # if not filename.endswith(".png"):
            #    continue
            path = imgPath + "/" + filename[0:11] + ".jpg"
            mapPath = activateMapPath + "/" + filename
            outPath = savePath + "/" + filename
            print(path)
            img = cv2.imread(path, 1)
            map = cv2.imread(mapPath, 0)
            heatMap = 0.5 * img + 0.5 * cv2.applyColorMap(map, cv2.COLORMAP_JET)
            cv2.imwrite(outPath, heatMap)



if __name__=="__main__":
    processor = ViewVOC2012()
    processor.getHeatMap("./test","./test1","./test2")
