import numpy as np
class perceptroDuality(object):
    def __init__(self,list):
        self.list=np.asarray(list)
        self.x=self.list[:,:2]
        self.y=self.list[:,2]
        self.a=np.zeros([3])
        self.b=np.zeros([1])

    def gramMatrix(self,x):
        return np.matmul(x,x.T)

    def updateWeights(self,yi,i):
        self.a[i]=self.a[i]+1
        self.b+=yi

    def perceptro_duality(self):
        update_times=0
        for epochs in range(20):                          #训练最多20代
            error_num = 0
            for i in range(len(self.x)):                  #每一代循环x
                temp_list=np.zeros_like(self.x.shape[1])  #计算误分数据,如果计算结果小于等于0就进行a,b的更新
                for j in range(len(self.a)):
                    temp_list=temp_list+self.a[j]*self.y[j]*self.x[j]
                result_point=self.y[i]*(np.matmul(temp_list,self.x[i])+self.b)
                if result_point<=0:
                    error_num+=1
                    update_times+=1
                    self.updateWeights(self.y[i],i)
                    print('Update times: %d;'%update_times,'Error value= x%d;'%(i+1),self.a,self.b)

            if error_num==0:
                break

    def perceptro_duality_two(self):
        update_times = 0
        gram=self.gramMatrix(self.x)       #先获取到x的内积
        for epochs in range(20):           # 训练最多20代
            error_num = 0
            for i in range(len(self.x)):   # 每一代循环x, 计算误分数据,如果计算结果小于等于0就进行a,b的更新
                result_point=0
                for j in range(len(self.a)):
                    result_point = result_point + self.a[j] * self.y[j] * gram[i][j]
                result_point=(result_point+self.b)*self.y[i]

                if result_point<= 0:
                    error_num += 1
                    update_times += 1
                    self.updateWeights(self.y[i], i)
                    print('Update times: %d;' % update_times, 'Error value= x%d;' % (i + 1), self.a, self.b)
            if error_num == 0:
                break


    def main(self):
        print(self.gramMatrix(self.x))


if __name__=='__main__':
    test=perceptroDuality([[3,3,1],[4,3,1],[1,1,-1]])
    test.perceptro_duality_two()                        #这两个方法实现结果一致
    # test.perceptro_duality()