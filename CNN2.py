# coding: UTF-8
from enum import Enum
import sys,os
sys.path.append(os.pardir)
import struct
import numpy as np
import mnist
import gc
#import matplotlib.pyplot as plt

"""
from dataset.mnist import load_mnist


def load_mnist(path, kind='train'):
        labels_path= os.path.join(path,'%s-labels.idx1-ubyte' % kind)
        images_path= os.path.join(path,'%s-images.idx3-ubyte' % kind)
        
        with open(labels_path, 'rb') as lbpath:
		magic,n=struct.unpack('>II',lbpath.read(8))
		labels=np.fromfile(lbpath, dtype=np.uint8)
        
        with open(images_path, 'rb') as imgpath:
		magic,num,row,cols=struct.unpack(">IIII", imgpath.read(16))
		images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
        
        return images,labels
"""
#def filter_show(w):
#	self.im2col(self.z[i-1].reshape(self.batchcount,self.channels[i],self.size[i][0],self.size[i][1])
	
def tanh(x):
	y=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	return y

def Dtanh(x):
	return 1-tanh(x)*tanh(x)

def relu(x):
	return np.maximum(0, x)

def Drelu(x):
	grad = x
	grad[x>=0] = 1
	grad[x<0] = 0
	return grad

def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T 

	x = x - np.max(x) # オーバーフロー対策
	return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t,axis):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
	if t.size == y.size:
		t = t.argmax(axis)
             
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def softmax_loss(X, t):
	y = softmax(X)
	return cross_entropy_error(y, t)


class Layers(Enum):
		INPUT= -1
		AFFINE_RELU = 0
		AFFINE_ONLY = 1
		CONV = 2
		POOLING = 3
		SOFTLOSS = 4

class Convolution: #畳み込み
	def __init__(self,unitcounts,batchcount,size,channels,laycount,laytypes,filtervals):
		self.unitcounts=unitcounts
		self.batchcount=batchcount
		self.size=size
		self.channels=channels
		self.xds=xds
		self.laycount=laycount
		self.W={}
		self.b={}
		self.laytypes=laytypes
		self.filtervals=filtervals
		self.z={}
		self.a={}
		self.y={}
		self.col={}
		self.argmax={}
		self.loss={}
		self.eps=0.05
		self.lam=0.001
		self.axis=1
		for i in range(1,self.laycount):
			if self.laytypes[i]==Layers.CONV:
				self.W[i]=np.random.randn(self.filtervals[2][i],self.channels[i],self.filtervals[0],self.filtervals[1])*norm
				self.b[i]=np.random.randn(self.size[i][0]*self.size[i][1],self.filtervals[2][i])*norm
			if laytypes[i]==Layers.AFFINE_RELU or laytypes[i]==Layers.AFFINE_ONLY or laytypes[i]==Layers.SOFTLOSS:
				self.W[i]=np.random.randn(self.unitcounts[i-1],self.unitcounts[i])*norm
				self.b[i]=np.random.randn(self.unitcounts[i])*norm
		
	def im2col(self, input_data, filter_h, filter_w, stride=1, pad=2):
		"""
        
		Parameters
		----------
		input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
		filter_h : フィルターの高さ
		filter_w : フィルターの幅
		stride : ストライド
		pad : パディング
        
		Returns
		-------
		col : 2次元配列
		"""
		N, C, H, W = input_data.shape
		out_h = (H + 2*pad - filter_h)//stride + 1
		out_w = (W + 2*pad - filter_w)//stride + 1
        
		img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
		col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        
		for y in range(filter_h):
			y_max = y + stride*out_h
			for x in range(filter_w):
				x_max = x + stride*out_w
				col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
		col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
		return col

	def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=2):
		"""
        
		Parameters
		----------
		col :
		input_shape : 入力データの形状（例：(10, 1, 28, 28)）
		filter_h :
		filter_w
		stride
		pad
        
		Returns
		-------
        
		"""
		N, C, H, W = input_shape
		out_h = (H + 2*pad - filter_h)//stride + 1
		out_w = (W + 2*pad - filter_w)//stride + 1
		col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        
		img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
		for y in range(filter_h):
			y_max = y + stride*out_h
			for x in range(filter_w):
				x_max = x + stride*out_w
				img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
		return img[:, :, pad:H + pad, pad:W + pad]
		
	def forward(self,xds):
		for i in range(0,self.laycount):
			if self.laytypes[i]==Layers.INPUT:
				self.z[i]=xds[0]

			if self.laytypes[i]==Layers.AFFINE_RELU:
				self.a[i]=np.dot(self.z[i-1],self.W[i])+self.b[i]
				self.z[i]=relu(self.a[i])
				
			if self.laytypes[i]==Layers.AFFINE_ONLY:
				self.a[i]=np.dot(self.z[i-1],self.W[i])+self.b[i]
				self.z[i]=self.a[i]
				        
			elif self.laytypes[i]==Layers.CONV:

				self.a[i]=np.dot(self.im2col(self.z[i-1].reshape(self.batchcount,self.channels[i],self.size[i][0],self.size[i][1]),self.filtervals[0],self.filtervals[1]),self.W[i].reshape(self.filtervals[2][i],-1).T)+np.tile(self.b[i],(self.batchcount,1))

				self.z[i]=relu(self.a[i])
				print(self.a[i].shape)
		        
			elif self.laytypes[i]==Layers.POOLING:
		            
				col=self.im2col((self.z[i-1].reshape(self.batchcount,self.size[i][0],self.size[i][1],self.channels[i])).transpose(0,3,1,2),self.filtervals[0],self.filtervals[1])
				col=col.reshape(-1,self.filtervals[0]*self.filtervals[1])
				self.argmax[i]=np.argmax(col,self.axis)
				self.a[i]=self.z[i]=np.max(col,self.axis).reshape(self.batchcount,self.size[i][0],self.size[i][1],self.channels[i]).transpose(0,3,1,2).reshape(self.batchcount,-1)
				
			elif self.laytypes[i]==Layers.SOFTLOSS:
				self.a[i]=np.dot(self.z[i-1],self.W[i])+self.b[i]
				self.z[i]=softmax(self.a[i])
				self.loss=cross_entropy_error(self.z[i],xds[1],self.axis)
				print("誤差:"+str(self.loss))
		              #  print(self.z[i].shape)
		              #  print(self.xds[1].shape)
		              #  print(softmax_loss(self.a[i],self.xds[1]))
				
	def backward(self,xds): # 逆伝播。計算は 1-1-7 項
		Delta={} # 各層のデルタ
		Delta[laycount - 1] = self.z[laycount - 1] - xds[1]; # 最終層のデルタを計算
		for i in range(laycount-2,0,-1):
# まずは活性化関数の微分(dfunc)をレイヤごとに定義しよう
			if self.laytypes[i]==Layers.AFFINE_RELU or self.laytypes[i]==Layers.CONV:
				dfunc=Drelu(self.a[i]) #ReLU が関わっている場合
			elif self.laytypes[i]==Layers.POOLING or self.laytypes[i]==Layers.AFFINE_ONLY or self.laytypes[i]==Layers.INPUT:
				dfunc=1 # 特にない場合。恒等写像と見て微分は 1
# 次に各層のデルタを計算。詳細は 1-1-7 項。注目する層の次の層の形で場合分けすることに注意
			if self.laytypes[i+1]==Layers.AFFINE_ONLY or self.laytypes[i+1]==Layers.AFFINE_RELU or self.laytypes[i+1]==Layers.SOFTLOSS: # 通常の場合
				Delta[i] = np.multiply(dfunc, np.dot(Delta[i + 1],self.W[i + 1].T))
			elif self.laytypes[i+1]==Layers.CONV: # 畳み込み層の場合
				bcol=self.col2im(Delta[i+1],[self.batchcount,self.channels[i+1],self.size[i+1][0],self.size[i+1][1]],\
				self.filtervals[0],self.filtervals[1])
				Delta[i]=np.multiply(dfunc,np.dot(bcol,self.W[i + 1].reshape(self.filtervals[2][i],-1).T))
				del bcol
				gc.collect()
			elif self.laytypes[i+1]==Layers.POOLING: # プーリング層 . これは極めて面倒
				pDelta=Delta[i+1].reshape(self.batchcount,self.channels[i+1],self.size[i+1][0],self.size[i+1][1]).transpose(0,2,3,1)
				dmax=np.zeros((pDelta.size, self.filtervals[0]*self.filtervals[1]))
				dmax[np.arange(self.argmax[i+1].size), self.argmax[i+1].flatten()] = pDelta.flatten()
				dmax=dmax.reshape(-1,self.channels[i+1]*self.filtervals[0]*self.filtervals[1])
				dmax=self.col2im(dmax,[self.batchcount,self.channels[i+1],self.size[i+1][0],self.size[i+1][1]],self.filtervals[0],self.filtervals[1])
				Delta[i]=dfunc*(dmax.transpose(0,2,3,1).reshape(-1,self.filtervals[2][i]))
				del pDelta
				gc.collect()
				del dmax
				gc.collect()
			del dfunc
			gc.collect()
		for i in range(1,laycount): # 重みの勾配をデルタに基づいて計算。前述のとおりデルタが分かれば計算は容易。
			if self.laytypes[i]!=Layers.INPUT and self.laytypes[i]!=Layers.POOLING: #INPUT と POOLING には重みはない
				if self.laytypes[i]==Layers.AFFINE_ONLY or self.laytypes[i]==Layers.AFFINE_RELU or self.laytypes[i]==Layers.SOFTLOSS:
					dw=self.eps*(np.dot(self.z[i-1].T,Delta[i]))+self.lam*self.W[i]
					self.W[i]=self.W[i]-(self.eps*dw) # ここが重み更新部分
				if self.laytypes[i]==Layers.CONV:
				#	print(Delta)
					dw=(np.dot(self.im2col(self.z[i-1].reshape(self.batchcount,self.channels[i],self.size[i][0],self.size[i][1]),self.filtervals[0],self.filtervals[1]).T,Delta[i]).T.reshape(self.filtervals[2][i],self.channels[i],self.filtervals[0],self.filtervals[1]))+self.lam*self.W[i]
					self.W[i]=self.W[i]-self.eps*dw # ここが重み更新部分
					del dw
					self.b[i]=np.tile(self.b[i],(self.batchcount,1))-self.eps*(Delta[i]+self.lam*np.tile(self.b[i],(self.batchcount,1)))
					self.b[i]=self.b[i][0:self.size[i][0]*self.size[i][1]]
					gc.collect()
				
		
(x_train,t_train),(x_test,t_test)=mnist.load_mnist(normalize=True, one_hot_label=True)
xds={}
tds={}
W={}
b={}
train_size=10000 # 訓練集合のサイズ
test_size=x_test.shape[0] # テスト集合のサイズ
batchcount=10 # ミニバッチの数

laycount=5 # ネットワークの層数
laytypes=[Layers.INPUT,Layers.CONV,Layers.POOLING,Layers.AFFINE_RELU,Layers.SOFTLOSS] # ネットワークの層の種類。仕様に対応
unitcounts=[784,15680,15680,100,10] # すべての層を普通のニューラルネットワークのユニットとみたときの、ユニット数
size=[[28,28],[28,28],[28,28],[-1,-1],[-1,-1]]#CNN に関わる層を画像的に捉えた時の、画像のサイズ。全結合層や出力層は、画像的処理から外れるので -1 として定義しない
# フィルタの大きさ
filterheight=5
filterwidth=5
norm=0.01
# チャネル数とフィルタ数。フィルタは 20 枚あるので、3 層目でチャネルが 20 倍になっていることに注意
channels=[1,1,20,-1,-1]
filternum=[-1,20,-1,-1,-1]
#CNN 本体の関数 Convolution を呼び出す
c=Convolution(unitcounts,batchcount,size,channels,laycount,laytypes,[filterheight,filterwidth,filternum])
#filter_show(c.W[1]) # 必要に応じてフィルタを表示
loss=0 # 誤差
f = open("result.txt","w")
w = open("weight.txt","w")
b = open("bias.txt","w")
# 訓練
for epoc in range(10): #10 × 1000 回学習。epoc はまとまった学習回数の単位で、ここでは 1epoc で 1000 回の学習を示す。
	loss=0
#if(epoc%5==0):filter_show(c.W[1])
	print("Epoc:"+str(epoc))
	for i in range(1000):
		batch_mask=np.random.choice(train_size,batchcount) # 訓練集合からランダムに選びバッチにする
		xds[0]=x_train[batch_mask]
		xds[1]=t_train[batch_mask]
		print(" 試行 :#"+str(i))
		c.forward(xds) # 順伝播
		c.backward(xds) # 順伝播
		loss+=c.loss
	loss/=1000.0
	f.write("##### Epoc "+str(epoc)+" #####"+"\n")
	f.write("loss "+str(loss)+"\n")
# テスト
coll=0# 正しく判定できた画像の数
c.batchcount=1 # テストなのでバッチは 1 でよい
for i in range(1000):
	f.write("Test #"+str(i+1)+" : Answer is "+str(t_test[i].argmax())+" CNN's Answer is "+str(c.forward([x_test[i].reshape(1,-1),t_test[i].reshape(1,-1)]).argmax())+"\n")
	if(t_test[i].argmax()==c.forward([x_test[i].reshape(1,-1),t_test[i].reshape(1,-1)]).argmax(1)):coll+=1
f.write("accuracy rate : "+str(coll)+" / 1000")
w.write(str(c.W))
b.write(str(c.b))
f.close()
w.close()
b.close()
print("Perfectly Done!")
