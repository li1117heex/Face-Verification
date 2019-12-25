import numpy as np

class Layer:#公共父类，函数无意义
    def initializer(self):
        pass
    ##def get_output_shape(self):
        ##pass
    def compute(self):
        pass
    def backpropagation(self):
        pass
    def save(self):
        pass
    def change_paras(self):
        pass

class Flatten(Layer):
    def initializer(self,input_shape,layer_index):
        self.index = layer_index
        self.output_shape = tuple([np.array(input_shape).prod()])
        self.input_shape = input_shape
    def compute(self,input:np.ndarray):
        return input.reshape(input.shape[0],-1)
    def backpropagation(self,output_deviation:np.ndarray,lr):
        return output_deviation.reshape(self.input_shape)

class Conv(Layer):

class MaxPooling(Layer):
    def __init__(self,pool_size=(2, 2), strides=None, padding='valid'):#stride=none represents no overlap pooling
        self.pool_size = pool_size
        if strides == None:
            self.strides = pool_size
        else:
            self.strides = strides
        self.padding = padding
    def initializer(self,input_shape,layer_index):
        self.index = layer_index
        if self.padding =="valid":
            self.output_shape = tuple([(input_shape[0]-self.pool_size[0])/self.strides[0]+1,(input_shape[1]-self.pool_size[1])/self.strides[1]+1,input_shape[-1]])
    def compute(self,input:np.ndarray):
        output = []
        input_argmax = []
        for sample in input:
            maxpool = np.zeros(self.output_shape)
            argmax = np.zeros(self.output_shape + (2))#打算计算方向传播用，写不下去
            row_current = 0
            row_index = 0
            while row_current+self.pool_size[0]<=input.shape[1]:
                column_current = 0
                column_index = 0
                while column_current+self.pool_size[1]<=input.shape[2]:
                    maxpool[row_index,column_index] =np.max(np.max(sample[row_current:row_current+self.pool_size[0],column_current:column_current+self.pool_size[1]],axis=0),axis=0)
                    column_current+=self.strides[1]
                    column_index+=1
                row_current +=self.strides[0]
                row_index +=1
            output.append(maxpool.copy())
        return np.concatenate(output,axis=0)
    def backpropagation(self,output_deviation:np.ndarray,lr):
        pass


class Dense(Layer):
    def __init__(self,units,activation=None, use_bias=True,kernel_initializer='zeros', bias_initializer='zeros'):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.type = 'Dense'
        self.output_shape = tuple([units])

    def initialier(self,input_shape,layer_index):#batch #初始化权值，index
        if (len(input_shape)!= 1):
            raise ValueError("Dense layer with input over 1 dim")
        self.index = layer_index
        self.kernel = eval(self.kernel_initializer)((input_shape[0],units))
        if(self.use_bias):
            self.bias = eval(self.bias_initializer)(units)

    ##def get_output_shape(self):
        ##return [self.units]

    def compute(self,input:np.ndarray):
        self.input = input#batch
        if(input.ndim!=2):
            raise ValueError("Dense layer with input over 1 dim")
        if(self.use_bias):
            activation_output, self.activation_diff = eval(self.activation)(self.kernel.dot(input.T) + self.bias[:,np.newaxis])
        else:
            activation_output, self.activation_diff = eval(self.activation)(self.kernel.dot(input.T))
            return activation_output

    def backpropagation(self,output_deviation:np.ndarray,lr):
        output_deviation2 = output_deviation.dot(self.activation_diff)
        kernel_deviation1 = -lr*self.input.sum(axis=0) #because input is a batch
        kernel_deviation2 = np.outer(output_deviation2,kernel_deviation1)
        self.kernel += kernel_deviation2 #(m,n)+(,n) is well defined
        if(self.use_bias):
            bias_deviation = -lr*output_deviation2
            self.bias += bias_deviation
        return output_deviation2.dot(self.kernel)

    def save(self):
        pass

#Activation = Enum("Activation",(relu))
def relu(x):
    return x * (x > 0),(x > 0) #google said this is faster than abs/maximun
#return result and diff

#initializer
def zeros(shape):
    return np.zeros(shape)

#loss

#return result and diff

#metrics
def acc(y_true, y_pred):
    return np.mean(np.round(y_pred) == y_true)#输入不需要round,经过测试，ytrue可以为bool