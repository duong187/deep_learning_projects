from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
from cv2 import cv2
import pickle
import os

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"

def validate_prediction(prediction,expected):
    expected=np.array(list(expected))
    prediction=np.array(prediction)
    comparision=expected==prediction
    result=comparision.all()
    return result

def resize_to_fit(image, width, height):
    """
    Điều chỉnh kích thước của hình ảnh image thành width x height
    Tham số:
    image: hình ảnh cần điều chỉnh kích thước
    width: chiều rộng muốn đạt được(đơn vị pixel)
    height: chiều cao muốn đạt được (đơn vị pixel)
    Trả về hình ảnh đã chỉnh sửa kích thước
    """

    # lấy giá trị các chiều của hình ảnh
    (h, w) = image.shape[:2]

    # Nếu chiều rộng lớn hơn chiều cao thì thực hiện thay đổi kích thước theo chiều rộng
    if w > h:
        image = imutils.resize(image, width=width)

    # Ngược lại, nếu chiều rộng nhỏ hơn chiều cao thì thay đổi kích thước theo chiều cao
    else:
        image = imutils.resize(image, height=height)

    # Xác định giá trị padding để hình ảnh đạt được những số chiều mong muốn
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Thực hiện padding
    # Dùng cv2.resize() để khắc phục những sai số về kích thước ảnh
    # có thể xuất hiện sau khi padding
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # trả về hình ảnh đã được thay đổi kích thước và padding
    return image

# Tải lại các nhãn (label) one-hot thu được từ bước train model
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Tải model đã huấn luyện
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
test_num=0.0
true_prediction_num=0.0
# loop over the image paths
for image_file in captcha_image_files:
    test_num+=1
    filename = os.path.basename(image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    # sử dụng openCV load hình ảnh
    # chuyển ảnh thành dạng xám(grayscale)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo viền cho bức ảnh
    # viền dạng BORDER_REPLICATE: sao chép các cạnh ngoài cùng của ảnh gốc, lấy các cạnh đó làm viền
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # thực hiện threshold
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # xác định đường viền bao quanh các vật thể (contour)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Kiểm tra phiên bản của openCV
    # Nếu là cv2 thì contour cần tìm ở contours[0]
    # Nếu là cv3 thì contour cần tìm ở contour [1]
    contours = contours[0] #if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # trả về 1 tam giác vuông bao quanh đường viền
        # x: hoành độ của đỉnh phía trên; y: tung độ của đỉnh phía trên; w: độ dài cạnh đáy; h:chiều cao tam giác
        (x, y, w, h) = cv2.boundingRect(contour)

         # So sánh tỷ lệ giữa độ dài đáy và độ cao để xác định các ký tự bị dính vào nhau
        # nếu w/h lớn hơn mọt mức nào đó nghĩa là contour tương ứng có 2 ký tự bị dính vào nhau
        if w / h > 1.25:
            # Cắt đôi cặp ký tự bị dính
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # Nếu không phải cặp ký tự dính nhau thì trả về các tọa độ ban đầu
            letter_image_regions.append((x, y, w, h))

   # Nếu số ký tự tìm thấy khác 4 nghĩa là ví dụ học không chuẩn, bỏ qua ví dụ học này
    if len(letter_image_regions) != 4:
        continue

   # sắp xếp các ký tự theo chiều tăng dần hoành độ x
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Tạo một hình ảnh đầu ra và một list để lưu các ký tự dự đoán
    output = cv2.merge([image] * 3)
    predictions = []

    # Duyệt qua tất cả các ký tự trong ảnh
    for letter_bounding_box in letter_image_regions:
        # Lưu lại tọa độ của ký tự đang xét
        x, y, w, h = letter_bounding_box

        # Trích xuất ảnh từ hình ảnh gốc, bổ sung lề có độ rộng 2 pixel
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        # Chuẩn hóa kích thước ảnh về 20x20
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Mô hình mạng neural đưa ra dự đoán
        prediction = model.predict(letter_image)

        # Giải mã biểu diễn one-hot của dự đoán, trả về dự đoán dưới dạng ký tự
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # Vẽ ký tự dự đoán
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if validate_prediction(prediction=predictions,expected=captcha_correct_text):
        true_prediction_num+=1
    # In dự đoán captcha dưới dạng text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    
    cv2.imshow("Output", output)
    cv2.waitKey()
print("The correct ratio of model's prediction is {}%".format(true_prediction_num/test_num*100))