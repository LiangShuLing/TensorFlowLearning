import numpy as np

'''
H_D 经验熵   一个数值
H_DA 特质A对数据集D的经验条件熵  用一个数组表示，比如[['年龄',0.01],['工作'，0.22]]
g_DA 信息增益 
'''
class desicion_making_tree(object):
    def __init__(self,list):
        super(desicion_making_tree, self).__init__()
        self.list=np.asarray(list)
        self.labels=np.asarray(['年龄','工作','房子','信贷'])

    def calcEntropy(self,dataSet):
        numEntries=len(dataSet)                       #数据集大小
        labelCounts={}                                  #用于记录类别以及个数
        for dataVec in dataSet:
            currentLabel=dataVec[-1]                    #取每个数据最后一个值为类别
            if currentLabel not in labelCounts.keys():  #如果label还没有保存至labelCounts里面就加入currentlabel，初始值为0
                labelCounts[currentLabel]=0
            labelCounts[currentLabel]+=1                #有了label的字典后，循环一次就计数一次，统计总数
        entropy=0.0
        for key in labelCounts:
            probability=float(labelCounts[key])/numEntries  #计算pi
            entropy-=probability*np.math.log(probability,2) #H_D  循环叠加计算-pi*log(pi)
        return entropy

    def splitData(self,data,axis,value):
        subDataset=[]
        for featureVector in data:
            if featureVector[axis]==value:
                reducedVector=featureVector[:axis]
                reducedVector=np.concatenate((reducedVector,featureVector[axis+1:]))  #搜索传入的axis列等于value的行，然后拼接除了传入的axis列
                subDataset.append(reducedVector)
        return subDataset

    def chooseBestFeature(self,dataSet):              #计算信息增益
        numerFeatures=len(dataSet[0])-1               #有多少个特征
        baseEntropy=self.calcEntropy(dataSet)         #计算经验熵
        bestInformationGain=0.0
        bestFeature=-1
        for i in range(numerFeatures):
            featureList=[feature[i] for feature in dataSet]  #循环取特征值
            uniqueVals=np.unique(featureList)                #去重，保留仅有的该特征下的取值
            newEntropy=0.0
            for value in uniqueVals:                         #循环每个特征值，计算每个特征对数据集的经验条件熵
                subDataset=self.splitData(dataSet,i,value)   #搜索传入的特征列并且等于value的数据集
                prob=len(subDataset)/float(len(dataSet))     #|Di|/|D|  子数据集长度与总数据集长度之比
                newEntropy+=prob*self.calcEntropy(subDataset) #计算特征A/B/C/D 对D的经验条件熵
                print(self.calcEntropy(subDataset))
            informationGain=baseEntropy-newEntropy            #A/B/C/D 对D的信息增益
            if informationGain>bestInformationGain:           #判断取最大信息增益下的特征
                bestInformationGain=informationGain
                bestFeature=i

            print(i, informationGain)

        return bestFeature



if __name__=='__main__':
    datalist = np.array([['青年', '否', '否', '一般', '否'],
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
                     ['老年', '否', '否', '一般', '否']])
    test=desicion_making_tree(datalist)
    print(test.chooseBestFeature(datalist))


