# Implementation of Biometric Logic - Automated Attendance System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Flask-red.svg" alt="Flask">
  <img src="https://img.shields.io/badge/Computer_Vision-OpenCV-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Database-Firebase-yellow.svg" alt="Firebase">
</div>

> **Academic Project Report** > School of Electrical and Electronic Engineering  
> Hanoi University of Science and Technology (HUST)

---

## 📑 Table of Contents
1. [1. Introduction](#1-introduction)
2. [2. Methodology](#2-methodology)
   - [2.1 Overall System Architecture](#21-overall-system-architecture)
   - [2.2 Database Schema and Storage](#22-database-schema-and-storage-design)
   - [2.3 Face Recognition Pipeline](#23-face-recognition-pipeline-development)
   - [2.4 Security and Access Control](#24-security-and-access-control)
3. [3. Project Implementation](#3-project-implementation)
   - [3.1 User Requirements & Survey](#31-user-requirement)
   - [3.2 System Block Design](#32-system-block-design)
4. [4. Installation & Setup](#4-installation--setup)
5. [5. Acknowledgement](#5-acknowledgement)

---

## 1. Introduction

### 1.1 Motivation
In current educational management systems, attendance tracking plays a vital role in evaluating student diligence and discipline. However, traditional methods such as manual roll calls or signing attendance sheets face significant challenges, including being time-consuming, prone to human error, and unable to effectively prevent fraud such as proxy attendance.

At large-scale institutions like Hanoi University of Science and Technology (HUST), managing thousands of students requires a more modern technological approach. This project was initiated to provide a modern, contactless attendance solution that reduces the administrative burden on instructors while offering convenience to students.

### 1.2 Objectives
* **Main Objective:** Design an interactive, web-based platform utilizing face recognition technology to automate identity verification and attendance record-keeping.
* **Specific Objectives:**
  * Build a core real-time image processing system via webcam using OpenCV.
  * Implement a cloud-based NoSQL database on Firebase to store student information and real-time attendance logs.
  * Design a user-friendly web interface (UI/UX) using Flask.
  * Develop a secure Teacher Login feature and provide Multi-class support.

---

## 2. Methodology

### 2.1 Overall System Architecture
The system is designed as a modular web-based application, integrating computer vision with cloud computing to automate attendance for the high-density classroom environment. The architecture consists of three primary components:

1. **Face Recognition Core:** The fundamental engine developed using Python and OpenCV. It handles the entire lifecycle of a visual input, from initial webcam capture to the final identification of a student.
2. **Web Interface (User Interface):** Developed using HTML, CSS, JavaScript, and Flask. It serves as the portal for student registration, displays real-time logs, and provides a secure dashboard for faculty.
3. **Cloud Infrastructure (Firebase):** Utilizes Firebase as a centralized NoSQL hub for managing student metadata and real-time attendance logs, while heavy image assets are maintained on local storage to prioritize processing speed.

*(Insert Figure 2.1: Overall System Architecture here)*
`![System Architecture](docs/images/figure_2_1.png)`

### 2.2 Database Schema and Storage Design
To manage the complex student hierarchy, the database is structured for efficiency:
* **User Metadata Collection (Firebase):** Stores `userID`, `name/email`, `embeddings` (numerical facial representation), `userType` (Student/Instructor), and `classes`.
* **Image Repository (Local):** Raw facial images are stored directly on the local SSD of the recognition client to eliminate network latency, named after the student's unique ID and synchronized with Firebase metadata.

### 2.3 Face Recognition Pipeline Development
The core processing logic follows a rigorous four-step pipeline to ensure accuracy under varying lecture hall conditions:

1. **Face Detection:** Utilizes **HOG (Histogram of Oriented Gradients) and Linear SVM** to scan video frames and isolate facial regions rapidly.
2. **Geometric Face Alignment:** Utilizing **68 facial landmarks**, the system scales and rotates the detected face to ensure the eyes and mouth are in fixed positions.
3. **Deep Feature Extraction (Embeddings):** The aligned face is passed through the **ResNet model** via the **DeepFace framework**, transforming the visual image into a 128-dimensional vector.
4. **Identity Matching:** The system identifies a student by calculating the similarity (distance) between the real-time embedding and stored embeddings. The closest match outputs the identified `FACE_ID`.

*(Insert Figure 2.2: Face Recognition Pipeline Development here)*
`![Pipeline](docs/images/figure_2_2.png)`

### 2.4 Security and Access Control
* **Secure Authentication:** A teacher login feature prevents unauthorized access.
* **Credential Hashing:** Password security is managed through a hashing mechanism.
* **Data Integrity:** Using the `Werkzeug` library, the system safely handles all file uploads to prevent malicious injections.

---

## 3. Project Implementation

### 3.1 User Requirement
To validate the practical necessity of this system, a pilot survey was conducted among 30 senior engineering students at SoICT (B1 Building labs & Ta Quang Buu Library). 
* **60%** reported current modules rely on manual methods.
* **80%** identified "time consumption" and "attendance fraud" as significant drawbacks.
* **90%** believed automated systems reduce administrative burdens and provide real-time, accurate data.
* **Conclusion:** 70% of respondents expressed a clear preference for switching to an automated biometric solution, justifying our development.

*(Insert Figure 3.1: Percentile surveyed by 30 random people here)*
`![Survey Chart](docs/images/figure_3_1.png)`

### 3.2 System Block Design
The system is structured into three primary operational blocks:
1. **Face Recognition System (Python/OpenCV):** Central logic component capturing frames, extracting biometric features, and performing matching.
2. **User Interface (HTML/CSS/JS/Flask):** Interaction point displaying Firebase data and sending user inputs (registration, class selection) to the server.
3. **Firebase Database:** Central management engine validating student status in real-time, interacting continuously with the local recognition client.

*(Insert Figure 3.2 and Figure 3.3: Main blocks and Flow chart here)*
`![Block Design](docs/images/figure_3_2.png)`

---

### 3.5. Step 5: Detail Block Design
The detailed operational flow of the system is designed to ensure a seamless transition from image capture to database logging. As illustrated in the flow chart (Figure 3.3), the process begins with system initialization, where the camera is activated and pre-trained models (HOG and ResNet) are loaded into memory.

Once a video frame is captured, the system applies the face detection algorithm. If a face is successfully detected, the geometric alignment and deep feature extraction processes are executed consecutively to generate a 128-dimensional embedding. This embedding is then compared against the local image repository using a distance metric. Upon a successful match (distance below the set threshold), the student's ID is retrieved, and the attendance log is instantaneously pushed to the Firebase Realtime Database, followed by a User Interface update. If no match is found or the face is unrecognized, the system simply continues to scan the subsequent frames.

### Figure 3.3: Flow chart
(Chèn ảnh Flow chart của nhóm bạn vào đây)

### 3.6. Step 6: Best alternatives selection
In the development of the "Implementation of Biometric Logic," the engineering team conducted a rigorous analysis of available technologies. The goal was to select the optimal combination of algorithms and system architecture that balances accuracy, processing speed (FPS), and resource efficiency.

### 3.6.1. Selection of Face Detection Algorithms
The core of the system relies on the ability to detect faces accurately before recognition. We compared two of the most popular computer vision techniques: Haar Cascade Classifiers and Histogram of Oriented Gradients (HOG)

---

## 4. Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/biometric-logic.git](https://github.com/your-username/biometric-logic.git)
cd biometric-logic
```
2. **Setup Virtual Environment & Install Dependencies:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. **Download Model Weights:**
Ensure shape_predictor_68_face_landmarks.dat is downloaded and placed in the appropriate models directory.

4. **Firebase Setup:**
Add your Firebase Service Account .json key to the project root and update configuration paths.

5. **Run System:**
```bash
python app.py
```
=======
# Face-Recognition-Attendance-System
This is my first project applying OpenCV to deal with the recognized attendance. 

