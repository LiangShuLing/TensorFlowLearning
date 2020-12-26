from StatisticalLearningMethod import Five_two_createTree
import numpy as np

class C_fourFive_generatingAlgorithm(Five_two_createTree.createTree):
    def __init__(self,dataList,labels):
        super(C_fourFive_generatingAlgorithm, self).__init__(dataList,labels)
        self.dataSet=dataList
        self.labels=labels

    '''
    重写特征选择的计算方法，用信息增益比来选择特征，g(D|A)/HA(D)
    '''
    def chooseBestFeature(self,dataSet):
        numerFeatures = len(dataSet[0]) - 1  # 有多少个特征
        baseEntropy = self.calcEntropy(dataSet)  # 计算经验熵H(D)
        bestInformationGain = 0.0
        bestFeature = -1
        for i in range(numerFeatures):
            featureList = [feature[i] for feature in dataSet]
            uniqueVals = np.unique(featureList)
            newEntropy = 0.0
            featureEntropy=0.0
            for value in uniqueVals:
                subDataset = self.splitData(dataSet, i, value)  # 搜索传入的特征列并且等于value的数据集
                prob = len(subDataset) / float(len(dataSet))  # |Di|/|D|  子数据集长度与总数据集长度之比
                featureVal_entropy=self.calcEntropy(subDataset)
                newEntropy += prob * featureVal_entropy  # 计算特征A/B/C/D 对D的经验条件熵  page-74
                featureEntropy+=featureVal_entropy       # HA(D)  page-76
            if featureEntropy==0.0:                      #如果获取得到的特质对于D的熵等于0，那么原则上信息增益比无穷大，因为其他信息增益比小于1，所以设置为10足够了
                informationGain=10.0
            else:
                informationGain = (baseEntropy - newEntropy)/featureEntropy  # A/B/C/D 对D的信息增益比，g(D|A)/HA(D)

            if informationGain > bestInformationGain:    # 判断取最大信息增益比下的特征
                bestInformationGain = informationGain
                bestFeature = i

        return bestFeature


if __name__=='__main__':
    datalist = [['青年', '否', '否', '一般', '否'],
                     ['青年', '否', '否', '好', '否'],
                     ['青年', '是', '否', '好', '是'],
                     ['青年', '是', '是', '一般', '是'],
                     ['青年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '一般', '否'],
                     ['中年', '否', '否', '好', '否'],
                     ['中年', '是', '是', '好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['中年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '非常好', '是'],
                     ['老年', '否', '是', '好', '是'],
                     ['老年', '是', '否', '好', '是'],
                     ['老年', '是', '否', '非常好', '是'],
                     ['老年', '否', '否', '一般', '否']]
    labels = ['年龄', '工作', '房子', '信贷']
    test=C_fourFive_generatingAlgorithm(datalist,labels)
    print(test.create_Tree(test.dataSet,test.labels))
