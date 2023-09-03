import os
import tensorflow as tf
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 데이터 폴더 경로
folder_path = 'D:\\buillding image\\Training\\concreate'

# 클래스 레이블 목록 가져오기
class_labels = os.listdir(folder_path)
num_classes = len(class_labels)
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
                    image = cv2.resize(image, (256, 256))  # 이미지 크기 조절
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

# Inception-v3 모델 불러오기 (사전 훈련된 가중치 사용)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# 새로운 분류 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 전체 모델 정의
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(x_train, y_train, batch_size=48,
                    epochs=250, validation_data=(x_val, y_val))

model.save('C:\\딥러닝\\inceptionv3_model6_batch_48.h5')

# 클래스 별 정확도 출력
class_accuracy = {}
for class_idx, class_label in enumerate(class_labels):
    class_mask = y_val == class_idx
    class_images = x_val[class_mask]
    class_labels_true = y_val[class_mask]
    class_labels_pred = np.argmax(model.predict(class_images), axis=1)
    class_accuracy[class_label] = np.mean(class_labels_true == class_labels_pred)

print("Accuracy for each class:")
for class_label, acc in class_accuracy.items():
    print(f"{class_label}: {acc:.2f}")

# 정확도 및 손실 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()