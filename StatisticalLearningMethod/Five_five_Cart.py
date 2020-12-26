import numpy as np

class cart_algorithm(object):
    def __init__(self,list):
        self.list=np.asarray(list)

    def splitData(self,dataList,axis,value):   #由于分割的数据集为连续的变量，所以采用大于等于方式来划分
        left_list=[]
        right_list=[]
        for data in dataList:
            if data[axis]<=value:
                left_list.append(list(data))
            else:
                right_list.append(list(data))
        return np.asarray(left_list),np.asarray(right_list)

    def square_error(self,list):
        square_value = 0.0
        if list.shape==[0,]:
            return square_value
        else:
            try:
                mean_value = np.mean(list[:, 1])
            except:
                print(list.shape)
        for value in list:
            square_value+=np.square(value[1]-mean_value)
        return square_value

    def chooseBestFeature(self,dataSet):
        SquareError_min=np.inf
        dataSet=np.sort(dataSet,axis=0)              #对数据集按照x大小排序,然后计算最小平方差后只需要知道哪个输入值就是
        bestSplitPoint=[]
        bestIndex=None
        for i in range(len(dataSet)-1):                #循环所有数据点，然后将其作为切分点，然后计算不同平方误差
            left,right=self.splitData(dataSet,0,dataSet[i][0])
            cureentSquareError=self.square_error(left)+self.square_error(right)
            if SquareError_min>=cureentSquareError: #计算平方差然后保存到一个list里面
                SquareError_min=cureentSquareError
                bestSplitPoint=dataSet[i]
                bestIndex=i
        # left_dataSet,right_dataSet=self.splitData(dataSet,0,dataSet[bestIndex][0])
        # left_c,right_c=np.mean(left_dataSet[:,1]),np.mean(right_dataSet[:,1])
        meanVal_bestSplitPoint=np.mean(dataSet[:,1])
        return bestSplitPoint,bestIndex,meanVal_bestSplitPoint

    def createRegressionTree(self,dataSet):
        if len(dataSet)==1:                #如果這時候數據集只有一個，那麽不再劃分直接輸出
            return dataSet[0][1]
        treeDic={}

        bestSplitPoint,bestIndex,val=self.chooseBestFeature(dataSet)
        treeDic["SplitValue"]=list(bestSplitPoint)
        treeDic["meanValue"]=val
        left_data,right_data=self.splitData(dataSet,0,dataSet[bestIndex][0])
        treeDic['left']=self.createRegressionTree(left_data)
        treeDic['right']=self.createRegressionTree(right_data)

        return treeDic
#由於該方法基本按照訓練數據集來進行的劃分，所以完全依賴訓練數據集的完整度，所以需要進行一定的優化

if __name__=='__main__':
    dataList=[[1,4.5],[2,4.75],[3,4.91],[4,5.34],[5,5.8],[6,7.05],[7,7.9],[8,8.23],[9,8.7],[10,9.0]]
    test=cart_algorithm(dataList)
    print(test.createRegressionTree(dataList))
    # left,right=test.splitData(test.list,0,5)
    # print(test.square_error(right))
    # print(test.square_error(left))
    # print(test.chooseBestFeature(test.list))



