import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train,x_test=x_train/255.0, x_test/255.0

y_train,y_test=to_categorical(y_train), to_categorical(y_test)

model=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax') ])


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))

loss,acc=model.evaluate(x_test,y_test)
print(loss,acc)

labels=["airplane", "automobile", "bird", "cat", "deer" , "dog" , "frog", "horse" , "ship", "truck"]

pred=model.predict(x_test[:3])

for i in range(3):
    fig, ax=plt.subplots(1,2,figsize=(10,2))
    ax[0].imshow(x_test[i])
    ax[0].axis('off')
    ax[1].barh(labels,pred[i],color='blue')
    ax[1].set_xlim(0,1)
    plt.tight_layout()
    plt.show()





import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models,initializers,regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train,x_test=x_train/255.0, x_test/255.0

y_train,y_test=to_categorical(y_train), to_categorical(y_test)

def create_model(initializer=None,dropout_rate=0.0,regularizer=None):
        model=models.Sequential([
            layers.Flatten(input_shape=(32,32,3)),
            layers.Dropout(dropout_rate),
            layers.Dense(512,activation='relu',kernel_initializer=initializer,kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(256,activation='relu',kernel_initializer=initializer,kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(128,activation='relu',kernel_initializer=initializer,kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(64,activation='relu',kernel_initializer=initializer,kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(32,activation='relu',kernel_initializer=initializer,kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(10,activation='softmax') ])
        return model


#base_model
model1=create_model()
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
base_train=model1.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
bloss,bacc=model1.evaluate(x_test,y_test)

#kaiming and xavier model
xavier=initializers.glorot_normal()

kaiming=initializers.he_normal()

xavier_model=create_model(initializer=xavier)
kaiming_model=create_model(initializer=kaiming)


xavier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
kaiming_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

xavier_train=xavier_model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
kaiming_train=kaiming_model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))

xloss,xacc=xavier_model.evaluate(x_test,y_test)
kloss,kacc=kaiming_model.evaluate(x_test,y_test)

print(xloss,xacc)
print(kloss,kacc)


#dropout
drop_model=create_model(dropout_rate=0.3)
drop_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
drop_train=drop_model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
dloss,dacc=drop_model.evaluate(x_test,y_test)

#l1 regularizer
l1_model=create_model(regularizer=regularizers.l1(0.01))
l1_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
l1_train=l1_model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
l1loss,l1acc=l1_model.evaluate(x_test,y_test)

#l2 regularizer
l2_model=create_model(regularizer=regularizers.l2(0.01))
l2_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
l2_train=l2_model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
l2loss,l2acc=l2_model.evaluate(x_test,y_test)

plt.plot(base_train.history['val_accuracy'],label='kaiming')
plt.plot(xavier_train.history['val_accuracy'],label='xavier')
plt.plot(kaiming_train.history['val_accuracy'],label='kaiming')
plt.plot(drop_train.history['val_accuracy'],label='kaiming')
plt.plot(l1_train.history['val_accuracy'],label='kaiming')
plt.plot(l2_train.history['val_accuracy'],label='kaiming')
plt.xlabel("epochs")
plt.ylabel("validation_accuracy")
plt.title("Weight  initializer adn regularizer")
plt.legend()
plt.show()





import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers,models,initializers,regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train,x_test=x_train/255.0, x_test/255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

y_train,y_test=to_categorical(y_train), to_categorical(y_test)


model=models.Sequential([
    layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')   ])




model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_train=model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
loss,acc=model.evaluate(x_test,y_test)



print(loss,acc)





import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16

(x_train,y_train),(x_test,y_test)= mnist.load_data()

npad=((0,0),(10,10),(10,10))

x_train=np.pad(x_train,npad,'constant',constant_values=0)
x_test=np.pad(x_test,npad,'constant',constant_values=0)

x_train=np.array([np.stack((img,img,img),axis=-1)for img in x_train])
x_test=np.array([np.stack((img,img,img),axis=-1) for img in x_test])

x_train,x_test=x_train/255.0, x_test/255.0

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

vgg_model=VGG16(weights='imagenet',include_top=False, input_shape=(48,48,3))

vgg_model.trainable=False

model=models.Sequential([
    vgg_model,
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
    ])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,batch_size=400,validation_data=(x_test,y_test))
loss,acc=model.evaluate(x_test,y_test)

print(loss,acc)






