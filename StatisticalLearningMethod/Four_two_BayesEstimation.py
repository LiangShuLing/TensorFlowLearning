import numpy as np
from StatisticalLearningMethod.Four_one_NavieBayes import navieBayes_classfication

class bayes_estimantion(navieBayes_classfication):             #继承上一个类，重写先验概率与条件概率计算即可
    def __init__(self,list):
        super(bayes_estimantion, self).__init__(list)
        self.k=len(self.y_val)
        self.lamb=1

    def prioriProbability(self):
        p_y=[]
        sum_y=[]
        for i in self.y_val:
            p_val=[i]
            len_i=np.sum(self.y==i)
            p_val.append((len_i+self.lamb)/(len(self.y)+self.k*self.lamb))         #加入修正的贝叶斯估计,注意加括号
            p_y.append(p_val)
            sum_y.append([i,len_i])
        return p_y,sum_y

    def conditionProbability(self):
        p_xy = []
        p_y, sum_y = self.prioriProbability()

        def CalculateProbability(x, index_x, y_val):
            for x_ in x:
                temp = []
                temp.append(x_)
                temp.append(y_val)
                temp.append((self.statisticNum(x_, index_x, y_val)+self.lamb) / (np.sum(self.y == y_val)+len(x)*self.lamb))  #重写条件概率计算
                p_xy.append(temp)

        for y_ in self.y_val:
            for index in range(len(self.list) - 1):
                CalculateProbability(np.unique(self.list[:, index:index + 1]), index, y_)

        return p_xy, p_y

if __name__=='__main__':
    test=bayes_estimantion([[1,'S',-1],
                                   [1,'M',-1],
                                   [1,'M',1],
                                   [1,'S',1],
                                   [1,'S',-1],
                                   [2,'S',-1],
                                   [2,'M',-1],
                                   [2,'M',1],
                                   [2,'L',1],
                                   [2,'L',1],
                                   [3,'L',1],
                                   [3,'M',1],
                                   [3,'M',1],
                                   [3,'L',1],
                                   [3,'L',-1]])
    result=test.classification([2,'S'])
    print(result)