import os
import os.path
from cv2 import cv2
import glob
import imutils


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# tạo một list với các phần tử là các file ảnh cần xử lý
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# duyệt qua tất cả các file hình ảnh
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # tên file có chứa nội dung của captcha
    # trích xuất nội dung text captcha từ tên file
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # sử dụng openCV load hình ảnh
    # chuyển ảnh thành dạng xám(grayscale)
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo viền cho bức ảnh
    # viền dạng BORDER_REPLICATE: sao chép các cạnh ngoài cùng của ảnh gốc, lấy các cạnh đó làm viền
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # thực hiện threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # xác định đường viền bao quanh các vật thể (contour)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kiểm tra phiên bản của openCV
    # Nếu là cv2 thì contour cần tìm ở contours[0]
    # Nếu là cv3 thì contour cần tìm ở contour [1]
    contours = contours[0] #if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    # Duyệt qua tất cả các đường viền
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

    # Lưu mỗi ký tự thành một ảnh
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Lưu lại các tọa độ của hình ảnh đang xét
        x, y, w, h = letter_bounding_box

        # trích xuất ký tự từ ảnh gốc, thêm margin có độ rộng 2
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        # Tạo folder chứa output
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # nếu đường dẫn chưa tồn tại thì tạo đường dẫn mới
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # nhập hình ảnh ký tự vào file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # đếm số ký tự thu được
        counts[letter_text] = count + 1
