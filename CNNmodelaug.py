import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터 폴더 경로 설정
folder_path = '/data/data/gcd1'

# 나머지 코드 유지

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

# 데이터 증강 생성기 설정
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

# CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (4, 4), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (4, 4), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (4, 4), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) // 32, epochs=20,
          validation_data=(x_val, y_val))

model.save('/data/data/gcd1/my_binary_model_with_augmentation.h5')