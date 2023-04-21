import cv2
import numpy as np
import os

"""
本文件用于对COCO2014数据集的预测结果进行可视化以及利用类激活图生成热图
"""
CLASS2COLOR = {1:(0, 0, 127), 2:(0, 0, 255), 3:(0, 63, 0), 4:(0, 63, 127), 5:(0, 63, 255),
         6:(0, 127, 0), 7:(0, 127, 63), 8:(0, 127, 127), 9:(0, 191, 0), 10:(0, 191, 63),
         11:(0, 191, 127), 12:(0, 191, 191), 13:(0, 191, 255), 14:(0, 255, 63),
         15:(63, 0, 0), 16:(63, 0, 63), 17:(63, 0, 127), 18:(63, 0, 255), 19:(63, 63, 0),
         20:(63, 63, 127), 21:(63, 63, 255), 22:(63, 127, 0), 23:(63, 127, 63),
         24:(63, 127, 255), 25:(63, 191, 0), 26:(63, 191, 63), 27:(63, 191, 127),
         28:(63, 255, 0), 29:(63, 255, 191), 30:(127, 0, 0), 31:(127, 0, 63), 32:(127, 0, 127),
         33:(127, 0, 191), 34:(127, 63, 0), 35:(127, 63, 255), 36:(127, 127, 0),
         37:(127, 127, 63), 38:(127, 127, 127), 39:(127, 127, 191), 40:(127, 191, 0),
         41:(127, 191, 127), 42:(127, 191, 191), 43:(127, 255, 0), 44:(127, 255, 63),
         45:(127, 255, 127), 46:(127, 255, 255), 47:(191, 0, 0), 48:(191, 0, 63),
         49:(191, 0, 255), 50:(191, 63, 0), 51:(191, 63, 255), 52:(191, 127, 0),
         53:(191, 127, 63), 54:(191, 127, 127), 55:(191, 127, 191), 56:(191, 127, 255),
         57:(191, 191, 0), 58:(191, 191, 63), 59:(191, 191, 127), 60:(191, 191, 191),
         61:(191, 191, 255), 62:(191, 255, 0), 63:(191, 255, 63), 64:(191, 255, 127),
         65:(191, 255, 191), 66:(191, 255, 255), 67:(255, 0, 0), 68:(255, 0, 127),
         69:(255, 63, 63), 70:(255, 63, 127), 71:(255, 63, 191), 72:(255, 63, 255),
         73:(255, 127, 0), 74:(255, 127, 63), 75:(255, 127, 127), 76:(255, 127, 191),
         77:(255, 127, 255), 78:(255, 191, 63), 79:(255, 191, 127), 80:(255, 191, 191)}

class ViewCOCO2014:
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
        for i in range(1,81):
            result[img == i] = CLASS2COLOR[i]
        return result

    # 对图片尺寸进行修改
    # imgPath为存放原始数据文件夹 savePath为存放输出数据文件夹 height为目标高度 weight为目标宽度
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
            path = imgPath + "/" + filename
            mapPath = activateMapPath + "/" + filename
            outPath = savePath + "/" + filename
            img = cv2.imread(path, 0)
            map = cv2.imread(mapPath, 1)
            heatMap = 0.5 * img + 0.5 * cv2.applyColorMap(map, cv2.COLORMAP_JET)
            cv2.imwrite(outPath, heatMap)





if __name__=="__main__":
    processor = ViewCOCO2014()
    processor.process("./SegmentationClass","./test")
    #img = cv2.imread("./SegmentationClass/COCO_train2014_000000000072.png", -1)
    #np.savetxt("a.txt",img)

