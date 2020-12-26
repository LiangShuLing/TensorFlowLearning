import numpy as np
class SolutioPercepron(object):
    '''
    感知机二分类
    Y标签为+1，-1
    损失函数：采用误分类点的损失值为-yi*(w*xi+b)
    采用梯度下降法：也就是每次输入一个训练集xi，然后循环查找是否有误分点，如果有就马上计算损失函数
       根据损失函数更新w,b
       w=w+n*yi+xi
       b=b+n*yi
    再次循环计算直至误分类点为0
    '''
    def __init__(self,list,lr=1):  #学习率默认为1
        self.list=np.asarray(list)        #list前两个数为坐标点，后一位数为标签
        self.w=np.zeros([2])
        self.b=np.zeros([1])
        self.lr=lr

    def innerProduct(self,x):
        matmul_val=np.matmul(x,self.w)+self.b
        pre_y=np.sign(matmul_val)
        return pre_y

    def updateWeigts(self,x,y):
        self.w=self.w+self.lr*y*x
        self.b=self.b+self.lr*y


    def percepron(self):
        iter_num=0              #统计weights更新次数
        for echo in range(10):
            error_val=0         #统计误分结果次数
            for values in self.list:
                x=values[0:2]
                y=values[2]
                pre_y=self.innerProduct(x)  #进行内积计算并计算预测结果
                if pre_y!=y:                #判断预测结果与实际结果，如果不等就更新权值，然后继续
                    self.updateWeigts(x,y)

                    error_val+=1
                    iter_num+=1
                    print('Updated the weigts: %d times' % iter_num)
                    print(values,self.w,self.b)
            if error_val==0:                   #如果本次循环中所有数据都正确划分就结束循环计算
                print('Now got the result: ')
                print(self.w,self.b)
                break


if __name__=='__main__':
    test=SolutioPercepron([[3,3,1],[4,3,1],[1,1,-1]])
    test.percepron()



