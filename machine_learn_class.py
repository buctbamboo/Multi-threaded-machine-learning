import warnings
warnings.filterwarnings("ignore")
import re
import math
from numpy import *
import threading
import pandas as pd
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import multiprocessing
import sys
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB #朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.neighbors import KNeighborsClassifier #K邻近
from sklearn.svm import SVC #支持向量机
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold

class MachineLearning:
    def __init__(self,xx,yy,number):
        self.number=number
        self.X=xx
        self.Y=yy
        
        self.num=0
        self.total=0;

        self.f_accurcy=[]
        self.f_precision=[]
        self.f_recall=[]
        self.f_fscore=[]
        
        self.accurcy=[]
        self.precision=[]
        self.recall=[]
        self.fscore=[]
        
        self.X_train=[]
        self.Y_train=[]
        self.X_test=[]
        self.Y_test=[]
        self.Y_predict=[]
        
        self.dtc=DecisionTreeClassifier()
        self.gb=GaussianNB()
        self.knn=KNeighborsClassifier()
        self.svm=SVC()
        self.skf= []
        
        self.i=0
        self._a=0
        self._b=0
        self._c=0
        self._d=0
        self.score=[]
        self.train_index=()
        self.test_index=()
        self.acc_score=0
        self.pre=[]
        self.rec=[]
        self.f1=[]
        self.support=[]

    def average(self,seq):
        self.num=0;
        self.total=0
        for item in seq:
            self.total+=item
            self.num+=1
        return self.total/self.num

    def Cross(self,dtc,num):
        self.skf = StratifiedKFold(self.Y, n_folds=num)
        self.accurcy=[]
        self.precision=[]
        self.recall=[]
        self.fscore=[]
        for self.train_index, self.test_index in self.skf:
            self.X_train,self.X_test=self.X[self.train_index],self.X[self.test_index]
            self.Y=np.array(self.Y).astype(int32)
            self.Y_train,self.Y_test=self.Y[np.array(self.train_index).astype(int32)],self.Y[np.array(self.test_index).astype(int32)]
            dtc.fit(self.X_train,self.Y_train)
            self.Y_predict=dtc.predict(self.X_test)
            self.acc_score=accuracy_score(self.Y_test,self.Y_predict)
            self.accurcy.append(self.acc_score)
            self.pre,self.rec,self.f1,self.support=precision_recall_fscore_support(self.Y_test,self.Y_predict)
            self.precision.append(self.average(self.pre))
            self.recall.append(self.average(self.rec))
            self.fscore.append(self.average(self.f1))
        return self.average(self.accurcy),self.average(self.precision),self.average(self.recall),self.average(self.fscore)

    def Clf_DTC(self):
        self.f_accurcy=[]
        self.f_precision=[]
        self.f_recall=[]
        self.f_fscore=[]
        for self.i in range(2,11):
            self._a,self._b,self._c,self._d=self.Cross(self.dtc,self.i)
            self.f_accurcy.append(self._a)
            self.f_precision.append(self._b)
            self.f_recall.append(self._c)
            self.f_fscore.append(self._d)
        return self.f_accurcy,self.f_precision,self.f_recall,self.f_fscore

    def Clf_GNB(self):
        self.f_accurcy=[]
        self.f_precision=[]
        self.f_recall=[]
        self.f_fscore=[]
        for self.i in range(2,11):
            self._a,self._b,self._c,self._d=self.Cross(self.gb,self.i)
            self.f_accurcy.append(self._a)
            self.f_precision.append(self._b)
            self.f_recall.append(self._c)
            self.f_fscore.append(self._d)
        return self.f_accurcy,self.f_precision,self.f_recall,self.f_fscore

    def Clf_KNN(self):
        self.f_accurcy=[]
        self.f_precision=[]
        self.f_recall=[]
        self.f_fscore=[]
        for self.i in range(2,11):
            self._a,self._b,self._c,self._d=self.Cross(self.knn,self.i)
            self.f_accurcy.append(self._a)
            self.f_precision.append(self._b)
            self.f_recall.append(self._c)
            self.f_fscore.append(self._d)
        return self.f_accurcy,self.f_precision,self.f_recall,self.f_fscore

    def Clf_SVM(self):
        self.f_accurcy=[]
        self.f_precision=[]
        self.f_recall=[]
        self.f_fscore=[]
        for self.i in range(2,11):
            self._a,self._b,self._c,self._d=self.Cross(self.svm,self.i)
            self.f_accurcy.append(self._a)
            self.f_precision.append(self._b)
            self.f_recall.append(self._c)
            self.f_fscore.append(self._d)
        return self.f_accurcy,self.f_precision,self.f_recall,self.f_fscore

    def Train(self):
    #决策树
        K_FOLD[self.number][0][0],K_FOLD[self.number][0][1],K_FOLD[self.number][0][2],K_FOLD[self.number][0][3]=self.Clf_DTC()

        #KNN
        K_FOLD[self.number][1][0],K_FOLD[self.number][1][1],K_FOLD[self.number][1][2],K_FOLD[self.number][1][3]=self.Clf_GNB()

        #朴素贝叶斯
        K_FOLD[self.number][2][0],K_FOLD[self.number][2][1],K_FOLD[self.number][2][2],K_FOLD[self.number][2][3]=self.Clf_KNN()

        #SVM
        K_FOLD[self.number][3][0],K_FOLD[self.number][3][1],K_FOLD[self.number][3][2],K_FOLD[self.number][3][3]=self.Clf_SVM()


    
def draw_pictures(pos):
    #绘制图表
    cols = ['blue','red','yellow', 'black']
    col_label = ["Decision Tree","KNN","Naive Bayesian","SVM"]
    samples=[0,1,2,3]
#    plt.xlim([2, 10])  
#    plt.ylim([0.9, 1])  
    plt.xlabel("Number of Folds")  
    plt.ylabel("Accuracy")  
    plt.grid()    
    for i, col in zip(samples, cols):
        plt.plot(x1, K_FOLD[pos][i][1],'o-',label=col_label[i], lw=3, color=col)
    plt.legend()
    str_name=pathfile[pos]+"\\picture11_"+str(pos)
    plt.savefig(str_name)
    plt.close()
#    plt.show()

def read_xy(X,Y,pathx,pathy):
    #获取X
    ff=open(pathx).readlines()
    for item in ff:
        X.append(np.array(list(item.split('\n')[0])).astype(np.int32))
    X=np.matrix(X).T

    #获取Y
    f=open(pathy).readlines()
    for item in f:
        Y.append(np.int32(item.split('\n')[0]))
    return X,Y

    
    
if __name__ == '__main__':
    
    #读取数据
    dataX=[[] for i in range(48)]
    dataY=[[] for i in range(48)]
    pathfile=[]
    K_FOLD=[[[[] for k in range(4)] for i in range(4)] for j in range(48)]
    x1 = np.linspace(2,10,9)
    x1=x1.tolist()
    _xfile="\\res_fault_version.out"
    _yfile="\\res_vector.out"

    #读取目录文件
    _pathfile=open("list.txt")
    for item in _pathfile:
        pathfile.append(item.split('\n')[0])
    
    
    lock = threading.Lock()
    
##    print(dataX.shape)
##    print(np.shape(dataY))

    threads = []
    
    for i in range(0,2):
        dataX[i],dataY[i]=read_xy(dataX[i],dataY[i],pathfile[i]+_xfile,pathfile[i]+_yfile)
        train_ans=MachineLearning(dataX[i],dataY[i],i)
        threads.append(threading.Thread(target=train_ans.Train))
    for t in threads:
        t.setDaemon(True)
        t.start()
    for i in range(0,2):
        threads[i].join()
    for i in range(0,2):
        draw_pictures(i)
        
        
        


        
    

    
