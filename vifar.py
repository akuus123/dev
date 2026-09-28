import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train,x_test=x_train/255.0,x_test/255.0
y_train,y_test=to_categorical(y_train),to_categorical(y_test)

hidden_units=[512,256,128]
activation='relu'

model=models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=64,
          validation_data=(x_test,y_test))

_,acc=model.evaluate(x_test,y_test)

print("Run 1:")
print("Hidden units:",hidden_units)
print("Activation:",activation)
print("Test accuracy:",round(acc*100,4))
print()

labels=["airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"]

pred=model.predict(x_train[:3])

for i in range(3):
    fig,ax=plt.subplots(1,2,figsize=(10,2))
    ax[0].imshow(x_train[i])
    ax[0].axis('off')
    ax[1].barh(labels,pred[i],color='blue')
    ax[1].set_xlim(0,1)
    plt.tight_layout()
    plt.show()
