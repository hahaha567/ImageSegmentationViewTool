import torch
import torch.nn.functional as F

"""
本文件用于在替换弱监督语义分割中的全局平均池化操作
"""

class SlectedAveragePool:
    def __init__(self):
        pass

    def pooling(self, feature, boxSize = 10, item = 1):
        batch, channel, height, weight = feature.shape
        featmap = feature.clone()
        for i in range(item):
            for b in range(batch):
                for c in range(channel):
                    index = torch.argmax(featmap[b, c])
                    h = index // height
                    w = index % weight
                    min_h = max(h - boxSize, 0)
                    min_w = max(w - boxSize, 0)
                    max_h = min(h + boxSize, height)
                    max_w = min(w + boxSize, weight)
                    featmap[b, c, min_h:max_h, min_w:max_w] = 0
        pred = F.avg_pool2d(feature[:, :-1], kernel_size=(height, weight), padding=0) - F.avg_pool2d(
            featmap[:, : -1], kernel_size=(height, weight), padding=0)
        return pred


if __name__=="__main__":
    processor = SlectedAveragePool()
    pred = processor.pooling(torch.randn(2,3,100,100))
    print(pred)

