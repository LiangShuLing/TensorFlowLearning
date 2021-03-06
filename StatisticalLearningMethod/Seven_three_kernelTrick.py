import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

'''
增加核技巧计算向量机,这里使用高斯径向基核(radial basis function)
K(x,z)=exp(-||x-z||^2/2o^2)

'''
def kernelTrans(X,Z,kTup):
    m,n=X.shape
    K=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':K=np.matmul(X,Z.T)   #线性支持向量机
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-Z
            K[j]=np.matmul(deltaRow,deltaRow.T)
        K=np.exp(K/(-1*kTup[1]**2))
    else: raise NameError('The Kernel is not recognized')
    return K

class optScruct(object):
    def __init__(self,dataSet,classLabels,C,toler,kTup):
        super(optScruct, self).__init__()
        self.X=np.mat(dataSet)
        self.labelMat=np.mat(classLabels).T
        self.C=C
        self.tol=toler
        self.m=dataSet.shape[0]
        self.alphas=np.mat(np.zeros([self.m,1]))
        self.b=0.0
        self.eCache=np.mat(np.zeros([self.m,2]))
        self.K=np.mat(np.zeros((self.m,self.m)))    #定义正定核,然后修改所有需要计算内积的参数
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)
        print(self.K.shape)


    def selectJrand(self,i,m):
        j=i
        while(j==i):
            j=int(np.random.uniform(0,m))  #随机产生一个数字，如果等于i那么继续产生，如果相等就返回这个数字
        return j

    def clipAlpha(self,aj,H,L):  #限定alpha的值大小，太大或者太小都返回其设定最大值、最小值
        if aj>H:
            aj=H
        elif aj<L:
            aj=L
        return aj

    def calEk(self,oS,k):
        x1 = np.multiply(oS.alphas, oS.labelMat).T
        fXk = np.matmul(x1, oS.K[:,k]) + oS.b
        Ek=fXk-float(oS.labelMat[k])
        return Ek

    def selectJ(self,i,oS,Ei):
        maxK=-1;maxDeltaE=0;Ej=0
        oS.eCache[i]=[1,Ei]
        validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]
        if len(validEcacheList)>1:
            for k in validEcacheList:
                if k==1:continue
                Ek=self.calEk(oS,k)
                deletaE=abs(Ei-Ek)
                if deletaE>maxDeltaE:
                    maxK=k;maxDeltaE=deletaE;Ej=Ek
                return maxK,Ej
        else:
            j=self.selectJrand(i,oS.m)
            Ej=self.calEk(oS,j)
        return j,Ej

    def updateEk(self,oS,k):
        Ek=self.calEk(oS,k)
        oS.eCache[k]=[1,Ek]

    def innerL(self,i,oS):
        Ei=self.calEk(oS,i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):  # 判断alpha是否需要优化
            # 如果上面i的alpha需要优化，那么随机选择第j个alpha。并计算fx与E值
            j,Ej = self.selectJ(i, oS,Ei)
            alphaI_old = oS.alphas[i].copy()
            alphaJ_old = oS.alphas[j].copy()
            if oS.labelMat[i] != oS.labelMat[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H: print("L=H");return 0
            eta = 2.0 * oS.K[i,j]-oS.K[i,i]-oS.K[j,j]
            if eta >= 0: print('eta>=0');return 0
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j], H, L)
            if (abs(oS.alphas[j]-alphaJ_old))<0.0000001: print('J not moving enough'); return 0
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJ_old - oS.alphas[j])
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) * oS.K[i,i] - \
                 oS.labelMat[j] * (oS.alphas[j] - alphaJ_old) * oS.K[i,j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) * oS.K[i,j] - \
                 oS.labelMat[j] * (oS.alphas[j] - alphaJ_old) * oS.K[j,j]

            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                print('updated BB1');b = b1  # 如果满足限制条件那么就获取到了所需要的b值，不满足取中值
            elif (0 < oS.alphas[j] and oS.C > oS.alphas[j]):
                print('updated BB2');oS.b = b2
            else:
                print('updated BB3');oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

def smop(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
        oS=optScruct(np.mat(dataMatIn),np.mat(classLabels),C,toler,kTup)
        iter=0
        entireSet=True;alphaPairsChanged=0
        calNum=0
        while(iter<maxIter) and (alphaPairsChanged>0 or entireSet):
            print('CalNumbers= %d'%calNum)
            alphaPairsChanged=0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged+=oS.innerL(i,oS)
                    print('FullSet, iter: %d i:%d, pris changed: %d'%(iter,i,alphaPairsChanged))
                    iter+=1
            else:
                nonBounds=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
                for i in nonBounds:
                    alphaPairsChanged+=oS.innerL(i,oS)
                    print('non-bound, iter: %d i:%d, pris changed: %d' % (iter, i, alphaPairsChanged))
                    iter+=1
            if entireSet: entireSet=False
            elif (alphaPairsChanged==0): entireSet=True
            print('Iteration numer: %d' % iter)
        return oS.b,oS.alphas

#构建径向基的测试函数
def testRbf(kl=1.3):
    #训练数据加载,然后训练计算出模型b,与alphas
    dataSet, labels = make_moons(50, noise=0.02)
    X_train, X_test, Y_train, Y_test = train_test_split(dataSet, labels, test_size=0.3)
    dataSet=np.mat(X_train)
    labelSet=np.mat(Y_train)
    print(labelSet.shape)


    b,alphas=smop(dataSet,labelSet,200,0.0001,10000,kTup=('rbf',kl))
    dataMat=np.mat(dataSet);labelMat=np.mat(labelSet).T
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=dataMat[svInd]
    labelSV=labelMat[svInd]
    print('There are %d Support Vector'%(sVs.shape[0]))

    def predicCal(dataX,dataY):
        m,n=dataX.shape
        errorCount=0
        for i in range(m):  #P142页，公式7.94
            kernelEval=kernelTrans(sVs,dataX[i,:],kTup=('rbf',kl))
            predicVal=kernelEval.T*np.multiply(labelSV,alphas[svInd])+b
            if np.sign(predicVal)!=np.sign(dataY[i]): errorCount+=1
        return errorCount/m

    trainErrorRate=predicCal(dataMat,labelMat)
    print('The training error rate is %f'%(trainErrorRate))

    testDataMat = np.mat(X_test)
    testLabelMat = np.mat(Y_test).T
    testErrorRate=predicCal(testDataMat,testLabelMat)
    print('The test error rate is %f' % (testErrorRate))


if __name__=='__main__':
    # dataSet=np.mat([[3,3],[4,3],[5,5],[1,1],[1,0],[0,1]])
    # label=np.mat([[1],[1],[1],[-1],[-1],[-1]])
    # b,alphas=smop(dataSet,label,0.6,0.00001,40)
    testRbf(kl=1.3)

    # print(b)
    # print(alphas)

