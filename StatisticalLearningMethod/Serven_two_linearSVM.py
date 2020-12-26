import numpy as np
import random

class linear_SVM(object):
    def __init__(self):
        super(linear_SVM, self).__init__()

    def selectJrand(self,i,m):
        j=i
        while(j==i):
            j=int(random.uniform(0,m))  #随机产生一个数字，如果等于i那么继续产生，如果相等就返回这个数字
        return j

    def clipAlpha(self,aj,H,L):  #限定alpha的值大小，太大或者太小都返回其设定最大值、最小值
        if aj>H:
            aj=H
        elif aj<L:
            aj=L
        return aj

    def simple_SMO(self,dataSet,classLabels,C,toler,maxIter):  #数据集，类标签，常数C,容错率和退出前最大循环次数
        dataMatrix=np.mat(dataSet)                             #转换为矩阵
        labelMat=np.mat(classLabels)                                   #转换为矩阵，然后转置.因为在输入之前已经转换，所以这里不做操作
        b=0.01                                                    #初始化b值为0
        m,n=dataMatrix.shape
        alphas=np.mat(np.random.random([m,1]))                                 #初始化alpha值，与输入训练集等长，一列
        iter=0
        while(iter<maxIter):
            alphaPairsChanged=0                                #用于记录alpha是否进行优化
            for i in range(m):
                x1=np.multiply(alphas,labelMat).T
                x2=np.matmul(dataMatrix,dataMatrix[i,:].T)
                fxi=np.matmul(x1,x2)+b

                Ei=fxi-float(labelMat[i])
                if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):  #判断alpha是否需要优化
                    #如果上面i的alpha需要优化，那么随机选择第j个alpha。并计算fx与E值
                    j=self.selectJrand(i,m)
                    temp1=np.multiply(alphas,labelMat).T
                    temp2=np.matmul(dataMatrix,dataMatrix[j,:].T)

                    fxj=np.matmul(temp1,temp2)+b
                    Ej=fxj-float(labelMat[j])

                    alphaI_old=alphas[i].copy()
                    alphaJ_old=alphas[j].copy()
                    if labelMat[i]!=labelMat[j]:
                        L=max(0,alphas[j]-alphas[i])
                        H=min(C,C+alphas[j]-alphas[i])
                    else:
                        L=max(0,alphas[j]+alphas[i]-C)
                        H=min(C,alphas[j]+alphas[i])
                    if L==H:print("L=H");continue
                    eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                    if eta>=0:print('eta>=0');continue
                    alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                    alphas[j]=self.clipAlpha(alphas[j],H,L)
                    # if (abs(alphas[j]-alphaJ_old))<0.0000001: print('J not moving enough'); continue

                    alphas[i]+=labelMat[j]*labelMat[i]*(alphaJ_old-alphas[j])
                    b1=b-Ei-labelMat[i]*(alphas[i]-alphaI_old)*dataMatrix[i,:]*dataMatrix[i,:].T-\
                       labelMat[j]*(alphas[j]-alphaJ_old)*dataMatrix[i,:]*dataMatrix[j,:].T
                    b2=b-Ej-labelMat[i]*(alphas[i]-alphaI_old)*dataMatrix[i,:]*dataMatrix[j,:].T-\
                       labelMat[j]*(alphas[j]-alphaJ_old)*dataMatrix[j,:]*dataMatrix[j,:].T

                    if (0<alphas[i]) and (C>alphas[i]): print('updated BB1');b=b1      #如果满足限制条件那么就获取到了所需要的b值，不满足取中值
                    elif (0<alphas[j] and C>alphas[j]): print('updated BB2');b=b2
                    else: print('updated BB3');b=(b1+b2)/2.0
                    alphaPairsChanged+=1
                    print('Iter: %d i:%d, paris changed %d'%(iter, i, alphaPairsChanged))
            if alphaPairsChanged==0: iter+=1                      #如果所有的数据i都不满足更新条件，也就是不再需要更新，那么再迭代结束，而此时后面的迭代理论上也不需要了
            else: iter=0                                          #如果还需要更新alpha，那么继续在第一迭代周期里面循环就是，所以iter>0就行，一样的结果
            print('Iteration numer: %d'%iter)
        return b,alphas



if __name__=='__main__':
    dataSet=np.mat([[3,3],[4,3],[5,5],[1,1],[1,0],[0,1]])
    label=np.mat([[1],[1],[1],[-1],[-1],[-1]])
    test=linear_SVM()
    b,alphas=test.simple_SMO(dataSet,label,0.6,0.00001,1)
    print(b)
    print(alphas)


    # alphas=np.mat(np.zeros_like(label.shape))
    # alpha=np.mat([[0],[0],[0],[0]])
    # data1=np.multiply(alpha,label).T
    # data=np.mat(np.matmul(dataSet,(dataSet[2,:].T))).T
    # if label[1]>0.6:
    #     print('hello')
    #
    #
    #
    # print(np.matmul(dataSet,dataSet[1:].T))
    # a=dataSet*dataSet[1,:].T
    # b=np.multiply(alpha,label).T
    # print(dataSet,dataSet[1,:].T)
    # print('-------')
    # print(dataSet*dataSet[1,:].T)
    #
    # print(np.multiply(alpha,label).T)
    #
    # Ei=b*a-float(label[1])
    # if label[1]*Ei<0.6:
    #     print('Ah ha')




    # labelVal=label[2]
    # print(labelVal)
    # fxi=data1*data+0.1
    # print(data1.shape,data.shape)
    # print(data1,data)
    # print(fxi-labelVal)
    # print('--------点乘，内积的运算以及不同----------')
    # a=np.array([1,2,3,4])
    # b=np.array([5,6,7,8])
    # print(np.matmul(a,b))
    # print(np.multiply(a,b))
    # print(np.multiply(a.T,b))
    # print(a.T)
