from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint
from keras.preprocessing import image
import numpy as np
import glob

inputsize=100
imgs=[]
iter=glob.glob('LFW/match pairs/*/*.jpg')[:2*inputsize].__iter__()
for file in iter:
    img1=image.img_to_array(image.load_img(file))
    img2=image.img_to_array(image.load_img(iter.__next__()))
    imgs.append(np.concatenate([img1,img2],axis=0)[np.newaxis,:])
iter=glob.glob('LFW/mismatch pairs/*/*.jpg')[:2*inputsize].__iter__()
for file in iter:
    img1=image.img_to_array(image.load_img(file))
    img2=image.img_to_array(image.load_img(iter.__next__()))
    imgs.append(np.concatenate([img1,img2],axis=0)[np.newaxis,:])
xtrain=np.concatenate(imgs,axis=0)/256.0
ytrain=np.concatenate([np.ones(inputsize),np.zeros(inputsize)])

patience = 100
logfilepath = "./log.csv"
trainedmmodelspath = "./trainedmodels/LeNet-5_1"

es = EarlyStopping("loss",0.1,patience)
rlr=ReduceLROnPlateau("loss",0.1,patience=int(patience/2),verbose=1)
csvl=CSVLogger(logfilepath,append=True)
modelnames=trainedmmodelspath+".{epoch:02d}-{acc:2f}.hdf5"
mcp=ModelCheckpoint(modelnames,'loss',1)
callbacks=[mcp,csvl,es,rlr]

model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(5,5),padding='valid',input_shape=(500,250,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
sgd=SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='mean_absolute_error',metrics=['acc'])
model.summary()
model.fit(xtrain,ytrain,batch_size=10,epochs=20,verbose=1,callbacks=callbacks,shuffle=True)

print(model.predict(xtrain,verbose=1))
