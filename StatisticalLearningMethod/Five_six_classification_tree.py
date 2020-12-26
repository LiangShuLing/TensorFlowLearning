import numpy as np
class classfificationTree(object):
    def __init__(self):
        pass
    def getSubdata(self,dataSet,data):
        subData=np.array([])
        for val in dataSet:
            print(val[0]==data)
            if val[0]==data:
                print(val[1])
                subData=np.append(subData,val[1])
        print(subData)
        return subData

    def gini(self,dataSet,feature_index):
        dataSet=np.asarray(dataSet)
        data_feature_class=np.hstack((np.expand_dims(dataSet[:, feature_index],axis=1),np.expand_dims(dataSet[:,-1],axis=1)))

        classVal_list=np.unique(data_feature_class[:,-1])
        featureVal_list=np.unique(data_feature_class[:,0])

        gini_val_list=np.array([])
        for feature_val in featureVal_list:
            gini_val=0.
            subData=self.getSubdata(data_feature_class,feature_val)
            for classVal in classVal_list:
                gini_val+=np.square(np.count_nonzero(subData==classVal)/len(subData))
            gini_val=(len(subData)/len(data_feature_class))*(1-gini_val)
            gini_val_list=np.append(gini_val_list,gini_val)

        return gini_val_list

if __name__=='__main__':
    test=classfificationTree()
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

    print(test.gini(datalist,2))