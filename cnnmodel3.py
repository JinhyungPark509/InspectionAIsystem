import os
import tensorflow as tf
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# 데이터 폴더 경로
folder_path = 'D:\\buillding image\\Training\\concreate'

# 클래스 레이블 목록 가져오기
class_labels = os.listdir(folder_path)

# 데이터와 레이블 초기화
data = []
labels = []

# 데이터 불러오기 및 전처리
for idx, label in enumerate(class_labels):
    class_path = os.path.join(folder_path, label)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if filename.lower().endswith('.jpg') and os.path.isfile(file_path):
                try:
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (224, 224))  # 이미지 크기 조절
                    data.append(image)
                    labels.append(idx)  # 이진 분류를 위해 클래스 인덱스를 레이블로 사용
                except Exception as e:
                    print(f'Error processing {file_path}: {str(e)}')

data = np.array(data) / 255.0  # 이미지를 0-1 범위로 정규화

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (4, 4), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (4, 4), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 시그모이드 활성화 함수
])

# RMSprop 옵티마이저 설정
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

model.save('C:\\딥러닝\\my_binary_model_with_rmsprop.h5')