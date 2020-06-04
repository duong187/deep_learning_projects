from cv2 import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import imutils



LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

def resize_to_fit(image, width, height):
    #Hàm thay đổi kích thước của hình ảnh image
    #Kích thước của image sau thay đổi là width x height

    #Xác định kích thước ban đầu của hình ảnh
    (h, w) = image.shape[:2]

    #Nếu chiều rộng lớn hơn chiều cao 
    #Thay đổi kích thước theo tỷ lệ chiều rộng
    if w > h:
        image = imutils.resize(image, width=width)

    # Ngược lại thì thay đổi kích thước theo tỷ lệ chiều cao
    else:
        image = imutils.resize(image, height=height)

    # Xác định giá trị padding phù hợp để đạt được số chiều mong muốn
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Padding
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # Trả lại bức ảnh đã được tiền xử lý
    return image

data = []
labels = []

# duyệt qua tất cả các hình ảnh đầu vào
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Đọc hình ảnh và chuyển sang định dạng grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Điều chỉnh kích thức hình ảnh thành 20x20
    image = resize_to_fit(image, 20, 20)

    # thêm chiều thứ 3 cho hình ảnh
    image = np.expand_dims(image, axis=2)

    # Xác định ký tự dựa trên tên file tương ứng
    label = image_file.split(os.path.sep)[-2]

    # Lưu lại hình ảnh ký tự và label của ký tự vào 2 list tương ứng
    data.append(image)
    labels.append(label)


# chuẩn hóa giá trị của từng pixel về khoảng [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Phân chia tập huấn luyện và tập kiểm thử
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Biểu diễn các label dưới dạng one-hot để làm việc với Keras
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Lưu lại ánh xạ từ nhãn (label) tới mã one-hot tương ứng của label đó
# Dùng để giải mã các mã one-hot thu được sau khi mô hình dự đoán
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Bắt đầu xây dựng mạng neural
model = Sequential()

# Layer tích chập đầu tiên
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
# MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer tích chập thứ hai
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
# Maxpooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Layer ẩn
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Layer đầu ra
model.add(Dense(32, activation="softmax"))

# Xây dựng mô hình Tensorflow
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Huấn luyện mạng neural 
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Lưu model đã huấn luyện
model.save(MODEL_FILENAME)
