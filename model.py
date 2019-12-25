import numpy as np
import layers

class Sequential(list):
    def __init__(self,input_shape):#与keras不同，输入shape在这里指定
        self.input_shape = input_shape
        self.output_shape = input_shape
    def add(self,layer:Layer):
        self.append(layer)
        layer.initializer(self.output_shape,len(self))
        self.output_shape = layer.output_shape#initializer后才有output_shape，对flatten来说
    def compile(self,lr,optimizer,loss=None, metrics=None):#暂时手动指定学习率
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        # metrics暂时只有一个
    def fit(self,x:np.ndarray=None,y:np.ndarray=None, batch_size=None, epochs=1, verbose=True, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True):
        #verbose我们应该只有2种
        if x.shape[0] != y.shape[0]:
            raise ValueError("input samples and labels number mismatch")
        if x.shape[1:]!=self.input_shape:
            raise ValueError("input samples shape mismatch")
        now = 0
        while now<epochs*x.shape[0]:
            batch_data, now = batch_generater(x, batch_size, epochs, 0)
            for layer in self:
                batch_data = layer.compute(batch_data)#在考虑应该是一次一个batch 还是一次一个，
            loss_value,diff = eval(self.loss)(batch_data,y)
            metrics_value = eval(self.metrics)(batch_data,y)
            if(verbose):
                print (now,"\tloss:",loss_value,"\tmetrics:",metrics_value)
            for layer in self[::-1]:#应该要改成把同一个batch不同sample的数据的微分分开，而不是原来想的加在一起
                diff = layer.backpropagation(diff,self.lr)
        batch_data = x
        for layer in self:
            batch_data = layer.compute(batch_data)
        loss_value, diff = eval(self.loss)(batch_data, y)
        metrics_value = eval(self.metrics)(batch_data, y)
        print ("finished,loss:", loss_value, "\tmetrics:", metrics_value)
        for layer in self:
            layer.save()
    def summary(self):
        for layer in self:
            print(layer.index,layer.type,"\t",layer.output_shape)
    def predict(self):
        pass
    def evaluate(self):
        pass

def batch_generater(x,batch_size,epoch,current):
    new=current+batch_size
    if(new>epoch*x.shape[0]):
        new = epoch*x.shape[0]
    if int(current/x.shape[0]) == int((new-1)/x.shape[0]):
        batch = x[current%x.shape[0]:new%x.shape[0]]
    else:
        batch = x[current%x.shape[0]:]
        temp = int(x/x.shape[0])+x.shape[0]
        while temp+x.shape[0]<new:
            batch=np.concatenate([batch,x],axis=0)
            temp+=x.shape[0]
        batch=np.concatenate([batch,x[:new%x.shape[0]]],axis=0)
    return batch,new
