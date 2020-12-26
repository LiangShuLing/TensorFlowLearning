import numpy as np
class navieBayes_classfication(object):
    def __init__(self,list):
        self.list=np.asarray(list)
        self.x1=self.list[:,:1]
        self.x1_val=np.unique(self.list[:,:1])
        self.x2=self.list[:,1:2]
        self.x2_val = np.unique(self.list[:,1:2])
        self.y=self.list[:, 2:3]
        self.y_val = np.unique(self.list[:, 2:3])


    def prioriProbability(self):
        p_y=[]                  #用于储存先验概率
        sum_y=[]
        for i in self.y_val:
            p_val=[i]           #先储存y的值，然后计算y的分布概率
            len_i=np.sum(self.y==i)
            p_val.append(len_i/len(self.y))
            p_y.append(p_val)
            sum_y.append([i,len_i]) #储存y的分布个数
        return p_y,sum_y

    def statisticNum(self,x,index_x,y):                 #统计条件满足下的训练数据数
        number=0
        for instance in self.list:
            if x==instance[index_x] and y==instance[2]:
                number+=1
        return number

    def conditionProbability(self):
        p_xy=[]
        p_y,sum_y=self.prioriProbability()
        def CalculateProbability(x,index_x,y_val):        #循环计算条件概率p(x|y)
            for x_ in x:
                temp=[]                                   #采用列表保存每一个条件变量
                temp.append(x_)
                temp.append(y_val)
                temp.append(self.statisticNum(x_,index_x,y_val)/np.sum(self.y==y_val))
                p_xy.append(temp)

        for y_ in self.y_val:                              #循环输入y value,x,index计算条件概率
            for index in range(len(self.list)-1):
                CalculateProbability(np.unique(self.list[:,index:index+1]),index,y_)

        return p_xy,p_y

    def classification(self,x):
        p_xy,p_y=self.conditionProbability()                 #获取条件概率,先验概率
        pyx=[]                                               #用于保存后验概率
        for py_value in p_y:                                 #循环不同分类下的先验概率
            p_temp=[py_value[0]]
            p_temp_value=py_value[1]                         #用于计算后验概率
            for index in range(len(x)):                      #循环待验证的输入特征,计算后验概率
                for pxy in p_xy:                             #循环条件概率分布，查找符合条件的条件概率
                    if str(x[index]) == str(pxy[0]) and str(py_value[0])==str(pxy[1]): #判断条件概率
                        print(x[index], pxy[0], py_value[0],pxy[1])
                        p_temp_value = p_temp_value * pxy[2]
            p_temp.append(p_temp_value)
            pyx.append(p_temp)

        #计算什么情况下的后验概率最大
        max_p=pyx[0]
        print(pyx)
        for value in pyx:
            if value[1]>=max_p[1]:
                max_p=value
        return max_p



if __name__=='__main__':
    test=navieBayes_classfication([[1,'S',-1],
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

