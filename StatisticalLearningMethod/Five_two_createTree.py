from StatisticalLearningMethod import Five_two_Decision_makingTree
import operator
import numpy as np
class createTree(Five_two_Decision_makingTree.desicion_making_tree):
    def __init__(self,datalist,labels):
        super(createTree, self).__init__(datalist)
        self.dataSet=np.asarray(datalist)
        self.labels = np.asarray(labels)

    def majorityCnt(self,classList):                 #多数表决模块
        classCount={}
        for vote in classList:                       #循环查看不同类，然后统计那种类的表决数量，决定到底属于哪种类别
            if vote not in classCount.keys(): classCount[vote]=0  #初始化classCount，如果不包含classList的类别就增加一个字典，key=类名，初始化为0
            classCount[vote]+=1                      #如果已经在里面，就增加该类别数为1

        sortedClassCount=sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse=True) #迭代classCount,并且按照第二值大小排序，然后倒序
        return sortedClassCount[0][0]                #然后返回第一个value

    def create_Tree(self,dataSet,labels):
        classList=[example[-1] for example in dataSet]     #取最后一列作为类别
        if classList.count(classList[0])==len(classList):  #如果所有类标签完全相同，也就是第一个类别的数目与整个类别列表长度相同,返回第一个类别就行
            return classList[0]                            #返回该完全划分的类别

        if len(dataSet[0])==1:                             #此时已经遍历完所有前面的特征，返回出现次数最多的类别
            return self.majorityCnt(classList)

        bestFeature=self.chooseBestFeature(dataSet)
        bestFetureLabel=labels[bestFeature]
        myTree={bestFetureLabel:{}}                                 #定义树形结构点,这样一个结点就已经完成运算了

        subLabels=np.delete(labels,bestFeature)                     #删除labels中最好的那个特征值对应的labels，然后递归计算其他特征
        fetaureValues=[example[bestFeature] for example in dataSet]  #得到最好特征下面对应的子数据集
        uniqueVals=np.unique(fetaureValues)
        for value in uniqueVals:                                     #递归循环计算最好特征，或者计算叶结点
            myTree[bestFetureLabel][value]=self.create_Tree(self.splitData(dataSet,bestFeature,value),subLabels)

        return myTree


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
    test=createTree(datalist,labels)
    print(test.create_Tree(test.dataSet,test.labels))


