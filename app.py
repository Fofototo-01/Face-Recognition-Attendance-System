from flask import Flask, render_template, request, redirect, url_for, Response, session
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import time
import numpy as np
import traceback
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from detection.face_matching import detect_faces, align_face
from detection.face_matching import extract_features
from utils.configuration import load_yaml

# Cấu hình hệ thống
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/recognized", exist_ok=True)

config_file_path = load_yaml("configs/database.yaml")
TEACHER_PASSWORD_HASH = config_file_path["teacher"]["password_hash"]

if not firebase_admin._apps:
    cred = credentials.Certificate(config_file_path["firebase"]["pathToServiceAccount"])
    firebase_admin.initialize_app(cred, {"databaseURL": config_file_path["firebase"]["databaseURL"]})

video = None
# Không dùng biến global match nữa, sẽ dùng session

app = Flask(__name__, template_folder="template")
# Secret key BẮT BUỘC phải có để dùng Session
app.secret_key = "F4_SO_HAND_SOME_SECRET_KEY_2026" 
app.config["UPLOAD_FOLDER"] = "static/images"

# --- HÀM AI ---
def match_with_database(img, database):
    # 1. Phát hiện tất cả các vật thể nghi là mặt
    faces = detect_faces(img)
    
    if len(faces) == 0:
        return None, "System could not find any face."

    # --- BỘ LỌC KÍCH THƯỚC (QUAN TRỌNG) ---
    valid_faces = []
    for (x, y, w, h) in faces:
        area = w * h
        if area > 6000: 
            valid_faces.append((x, y, w, h))

    # 2. Xử lý sau khi lọc
    if not valid_faces:
        return None, "No valid face detected (Too small or obscure)."

    # 3. Chọn khuôn mặt TO NHẤT trong số những mặt hợp lệ
    # (Để chắc chắn bắt đúng người ngồi gần cam nhất)
    x, y, w, h = max(valid_faces, key=lambda rect: rect[2] * rect[3])
    
    # Vẽ khung đỏ CHỈ lên khuôn mặt đã chọn 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imwrite("static/recognized/recognized.png", img)

    try:
        # Căn chỉnh và cắt khuôn mặt
        aligned_face = align_face(img, (x, y, w, h))
        
        # Trích xuất đặc trưng
        try:
            embedding_obj = extract_features(aligned_face, enforce_detection=False)
        except TypeError:
            embedding_obj = extract_features(aligned_face)

        current_embedding = np.array(embedding_obj[0]["embedding"])

        # So sánh với Database
        min_distance = 100 
        best_match_name = None
        THRESHOLD = 0.45 # Độ khó tính của AI 

        for name, db_embedding in database.items():
            if not db_embedding: continue
            
            # Tính toán khoảng cách
            dist = 1 - (np.dot(current_embedding, np.array(db_embedding)) / (np.linalg.norm(current_embedding) * np.linalg.norm(np.array(db_embedding))))
            
            if dist < min_distance:
                min_distance = dist
                best_match_name = name

        # Kết luận
        if min_distance < THRESHOLD and best_match_name:
            return best_match_name, f"Match found ({min_distance:.2f})"
        else:
            return None, "Unknown Person (Face detected but not recognized)"
            
    except Exception:
        traceback.print_exc()
        return None, "Error processing face image"
# --- ROUTES ---
@app.route("/")
def home():
    # Xóa sạch session cũ khi về trang chủ để tránh nhầm lẫn
    session.pop('user_name', None)
    return render_template("home.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_frames():
    global video
    if video is None or not video.isOpened():
        video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success: break
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# --- XỬ LÝ ĐIỂM DANH (RECOGNIZE) ---
@app.route("/recognize", methods=["POST"])
def recognize():
    global video
    if video is None or not video.isOpened(): video = cv2.VideoCapture(0); time.sleep(0.5)
    ret, frame = video.read()
    
    if ret:
        ref = db.reference("Students")
        data = ref.get()
        if not data: return redirect(url_for("select_class"))

        database = {}
        items = data if isinstance(data, list) else data.values()
        for val in items:
            if val and isinstance(val, dict) and "name" in val:
                database[val["name"]] = val["embeddings"]

        name, msg = match_with_database(frame, database)
        
        # LƯU VÀO SESSION (Bộ nhớ tạm của trình duyệt)
        if name:
            session['user_name'] = name 
            return redirect(url_for("select_class"))
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            url = url_for("static", filename="recognized/recognized.png", v=timestamp)
            return render_template("result.html", 
                                   status_class="bg-danger", 
                                   title="STUDENT NOT FOUND", 
                                   message="Who are you?", 
                                   detail=msg, 
                                   image_url=url)
    
    return redirect(url_for("home"))

# --- TRANG CHỌN LỚP (SỬA LỖI TRÔI TRANG) ---
@app.route("/select_class", methods=["GET", "POST"])
def select_class():
    # 1. Kiểm tra xem người dùng đã được nhận diện chưa?
    current_user = session.get('user_name') # Lấy tên từ session
    
    if not current_user:
        # Nếu chưa nhận diện mà cố vào trang này -> Đuổi về trang chủ
        return redirect(url_for("home"))

    # 2. Nếu là POST (Người dùng bấm nút Confirm)
    if request.method == "POST":
        selected_class = request.form.get("classes")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        url = url_for("static", filename="recognized/recognized.png", v=timestamp)
        
        # Tìm thông tin trong DB để check lớp
        ref = db.reference("Students")
        data = ref.get()
        target_key = None
        student_info = None

        items = data if isinstance(data, list) else data.items()
        
        # Logic tìm người (hỗ trợ cả List và Dict)
        if isinstance(data, list):
            for idx, val in enumerate(data):
                if val and val.get("name") == current_user:
                    target_key = str(idx); student_info = val; break
        else:
            for key, val in data.items():
                if val and val.get("name") == current_user:
                    target_key = key; student_info = val; break
        
        # Check lớp đăng ký
        if student_info:
            registered = student_info.get("classes", {})
            if selected_class not in registered:
                return render_template("result.html", 
                                       status_class="bg-warning", 
                                       title="WRONG CLASS", 
                                       message=f"Hi, {current_user}", 
                                       detail=f"You are NOT registered for {selected_class}.", 
                                       image_url=url)
            
            # Cộng điểm danh
            cnt = registered.get(selected_class, 0)
            ref.child(f"{target_key}/classes/{selected_class}").set(int(cnt) + 1)
            
            return render_template("result.html", 
                                   status_class="bg-success", 
                                   title="SUCCESS", 
                                   message=f"Welcome, {current_user}!", 
                                   detail=f"Checked in: {selected_class}", 
                                   image_url=url)
        
        return "Database Error"

    # 3. Nếu là GET (Mới vào trang): HIỂN THỊ MENU CHỌN LỚP
    # Code cũ bị trôi vì thiếu đoạn return này
    return render_template("select_class.html", student_name=current_user)


# --- CÁC ROUTE KHÁC (GIỮ NGUYÊN NHƯNG SỬA LỖI METHOD) ---
@app.route("/capture", methods=["POST"])
def capture():
    # (Giữ nguyên logic capture cũ của bạn)
    global video
    if video is None or not video.isOpened(): video = cv2.VideoCapture(0); time.sleep(0.5)
    ret, frame = video.read()
    if ret:
        ref = db.reference("Students")
        data = ref.get()
        # Logic ID đơn giản hóa
        next_id = int(time.time())
        filename = f"{next_id}.png"
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], filename), frame)
        session['temp_filename'] = filename # Lưu tạm tên file
    return redirect(url_for("add_info"))

@app.route("/add_info")
def add_info(): return render_template("add_info.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if f:
        filename = f"upload_{int(time.time())}.png"
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        session['temp_filename'] = filename
        return redirect(url_for("add_info"))
    return "Upload failed"

@app.route("/submit_info", methods=["POST"])
def submit_info():
    # Lấy tên file từ session 
    filename = session.get('temp_filename')
    if not filename: return "No image found. Capture again."
    
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = cv2.imread(path)
    
    try:
        try: emb = extract_features(img, enforce_detection=False)
        except: emb = extract_features(img)
        embedding_list = emb[0]["embedding"]
    except: return "Cannot extract face. Try again."

    ref = db.reference("Students")
    user_data = {
        "name": request.form.get("name"),
        "email": request.form.get("email"),
        "userType": request.form.get("userType"),
        "classes": {c: 0 for c in request.form.getlist("classes")},
        "embeddings": embedding_list
    }
    ref.child(str(int(time.time()))).set(user_data)
    return redirect(url_for("success", filename=filename))

@app.route("/success/<filename>")
def success(filename):
    url = url_for("static", filename="images/" + filename)
    return render_template("success.html", image_url=url)


@app.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        password = request.form.get("password")
        # Kiểm tra mật khẩu
        if check_password_hash(TEACHER_PASSWORD_HASH, password):
            session['is_admin'] = True # Cấp quyền Admin
            return redirect(url_for("attendance")) # Chuyển hướng sang trang danh sách
        else:
            return render_template("teacher_login.html", error="Wrong Password")
    
    return render_template("teacher_login.html")

@app.route("/attendance")
def attendance():
    # Bảo mật: Nếu chưa đăng nhập thì đuổi về trang login
    if not session.get('is_admin'):
        return redirect(url_for("teacher_login"))

    ref = db.reference("Students")
    data = ref.get()
    
    student_list = []
    
    # Xử lý dữ liệu từ Firebase (Chuẩn hóa thành List để dễ hiển thị)
    if data:
        # Nếu là List (do ID dạng 0,1,2...)
        if isinstance(data, list):
            for val in data:
                if val: student_list.append(val)
        # Nếu là Dict (do ID dạng timestamp...)
        elif isinstance(data, dict):
            for key, val in data.items():
                if val: student_list.append(val)

    return render_template("attendance.html", students=student_list)

# Route logout
@app.route("/logout")
def logout():
    session.pop('is_admin', None)
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)