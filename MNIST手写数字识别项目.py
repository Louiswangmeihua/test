from keras.datasets import mnist
from keras.utils import to_categorical
(X_train,y_train),(x_test,y_test)=mnist.load_data()

X_train=X_train/255.0
x_test=x_test/255.0
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense,Flatten
from  keras.optimizers import RMSprop #引入梯度下降优化算法

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=64,activation=("relu")))
model.add(Dense(units=32,activation=("relu")))
model.add(Dense(units=32,activation=("relu")))
model.add(Dense(units=10,activation=('softmax')))

#对模型进行编译，指定损失函数
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=RMSprop())
#对模型进行训练
model.fit(X_train,
          y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.2)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)