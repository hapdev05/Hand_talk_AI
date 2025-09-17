import cv2
import mediapipe as mp
import json
import numpy as np

# Khởi tạo MediaPipe
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Cấu hình đường dẫn
input_video = "anvung.mp4"  # Video đầu vào
output_video = "pose_visualization.mp4"  # Video đầu ra với landmarks
output_json = "pose_data_with_visualization.json"  # JSON với dữ liệu chi tiết

print("🎬 Tạo video visualization với MediaPipe...")
print(f"📹 Input: {input_video}")
print(f"🎥 Output video: {output_video}")
print(f"📄 Output JSON: {output_json}")

def draw_detailed_landmarks(image, results):
    """Vẽ landmarks chi tiết như trong hình mẫu"""
    annotated_image = image.copy()
    
    # Vẽ pose landmarks với style đẹp
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Vẽ face landmarks với điểm nhỏ
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 256, 121), thickness=1))
    
    # Vẽ left hand với màu riêng
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 255), thickness=3, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 255), thickness=2))
    
    # Vẽ right hand với màu riêng
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 255), thickness=3, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 0, 255), thickness=2))
    
    return annotated_image

def add_info_overlay(image, frame_count, total_frames, fps, landmarks_data):
    """Thêm thông tin overlay lên video"""
    height, width = image.shape[:2]
    
    # Tạo overlay trong suốt
    overlay = image.copy()
    
    # Thông tin frame
    cv2.putText(overlay, f"Frame: {frame_count}/{total_frames}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(overlay, f"Time: {frame_count/fps:.2f}s", 
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Thông tin detection
    y_offset = 120
    detections = [
        ("Pose", landmarks_data['pose']['landmarks']),
        ("Face", landmarks_data['face']),
        ("L.Hand", landmarks_data['left_hand']['landmarks']),
        ("R.Hand", landmarks_data['right_hand']['landmarks'])
    ]
    
    for name, data in detections:
        status = "✓" if data else "✗"
        color = (0, 255, 0) if data else (0, 0, 255)
        count = len(data) if data else 0
        
        cv2.putText(overlay, f"{name}: {status} ({count})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 35
    
    # Thông tin chi tiết ngón tay (nếu có)
    if landmarks_data['left_hand']['landmarks']:
        y_offset += 20
        cv2.putText(overlay, "Left Hand Fingers:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 25
        
        for finger_name, finger_data in landmarks_data['left_hand']['fingers'].items():
            cv2.putText(overlay, f"  {finger_data['name_vi']}: {len(finger_data['joints'])} joints", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
    
    if landmarks_data['right_hand']['landmarks']:
        y_offset += 10
        cv2.putText(overlay, "Right Hand Fingers:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 25
        
        for finger_name, finger_data in landmarks_data['right_hand']['fingers'].items():
            cv2.putText(overlay, f"  {finger_data['name_vi']}: {len(finger_data['joints'])} joints", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_offset += 20
    
    # Logo/watermark
    cv2.putText(overlay, "MediaPipe Pose Analysis", 
               (width - 400, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return overlay

def extract_detailed_landmarks(results):
    """Trích xuất landmarks chi tiết (copy từ script chính)"""
    # Hand landmarks mapping
    hand_landmark_names = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    data = {
        "pose": {"landmarks": [], "detailed": {}},
        "left_hand": {"landmarks": [], "detailed": {}, "fingers": {}},
        "right_hand": {"landmarks": [], "detailed": {}, "fingers": {}},
        "face": []
    }
    
    # Pose landmarks
    if results.pose_landmarks:
        data["pose"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)] 
                                   for lm in results.pose_landmarks.landmark]
    
    # Left hand landmarks
    if results.left_hand_landmarks:
        data["left_hand"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                                        for lm in results.left_hand_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            landmark_name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"hand_landmark_{i}"
            data["left_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay
        data["left_hand"]["fingers"] = {
            "thumb": {
                "name_vi": "Ngón cái",
                "joints": {
                    "cmc": {"name_vi": "Khớp gốc", "coordinates": data["left_hand"]["detailed"]["thumb_cmc"]["coordinates"]},
                    "mcp": {"name_vi": "Khớp giữa", "coordinates": data["left_hand"]["detailed"]["thumb_mcp"]["coordinates"]},
                    "ip": {"name_vi": "Khớp đầu", "coordinates": data["left_hand"]["detailed"]["thumb_ip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["left_hand"]["detailed"]["thumb_tip"]["coordinates"]}
                }
            },
            "index": {
                "name_vi": "Ngón trỏ",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["left_hand"]["detailed"]["index_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["left_hand"]["detailed"]["index_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["left_hand"]["detailed"]["index_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["left_hand"]["detailed"]["index_tip"]["coordinates"]}
                }
            },
            "middle": {
                "name_vi": "Ngón giữa",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["left_hand"]["detailed"]["middle_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["left_hand"]["detailed"]["middle_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["left_hand"]["detailed"]["middle_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["left_hand"]["detailed"]["middle_tip"]["coordinates"]}
                }
            },
            "ring": {
                "name_vi": "Ngón áp út",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["left_hand"]["detailed"]["ring_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["left_hand"]["detailed"]["ring_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["left_hand"]["detailed"]["ring_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["left_hand"]["detailed"]["ring_tip"]["coordinates"]}
                }
            },
            "pinky": {
                "name_vi": "Ngón út",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["left_hand"]["detailed"]["pinky_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["left_hand"]["detailed"]["pinky_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["left_hand"]["detailed"]["pinky_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["left_hand"]["detailed"]["pinky_tip"]["coordinates"]}
                }
            }
        }
    
    # Right hand landmarks (tương tự)
    if results.right_hand_landmarks:
        data["right_hand"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                                         for lm in results.right_hand_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            landmark_name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"hand_landmark_{i}"
            data["right_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay
        data["right_hand"]["fingers"] = {
            "thumb": {
                "name_vi": "Ngón cái",
                "joints": {
                    "cmc": {"name_vi": "Khớp gốc", "coordinates": data["right_hand"]["detailed"]["thumb_cmc"]["coordinates"]},
                    "mcp": {"name_vi": "Khớp giữa", "coordinates": data["right_hand"]["detailed"]["thumb_mcp"]["coordinates"]},
                    "ip": {"name_vi": "Khớp đầu", "coordinates": data["right_hand"]["detailed"]["thumb_ip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["right_hand"]["detailed"]["thumb_tip"]["coordinates"]}
                }
            },
            "index": {
                "name_vi": "Ngón trỏ",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["right_hand"]["detailed"]["index_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["right_hand"]["detailed"]["index_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["right_hand"]["detailed"]["index_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["right_hand"]["detailed"]["index_tip"]["coordinates"]}
                }
            },
            "middle": {
                "name_vi": "Ngón giữa",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["right_hand"]["detailed"]["middle_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["right_hand"]["detailed"]["middle_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["right_hand"]["detailed"]["middle_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["right_hand"]["detailed"]["middle_tip"]["coordinates"]}
                }
            },
            "ring": {
                "name_vi": "Ngón áp út",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["right_hand"]["detailed"]["ring_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["right_hand"]["detailed"]["ring_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["right_hand"]["detailed"]["ring_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["right_hand"]["detailed"]["ring_tip"]["coordinates"]}
                }
            },
            "pinky": {
                "name_vi": "Ngón út",
                "joints": {
                    "mcp": {"name_vi": "Khớp gốc", "coordinates": data["right_hand"]["detailed"]["pinky_mcp"]["coordinates"]},
                    "pip": {"name_vi": "Khớp giữa", "coordinates": data["right_hand"]["detailed"]["pinky_pip"]["coordinates"]},
                    "dip": {"name_vi": "Khớp đầu", "coordinates": data["right_hand"]["detailed"]["pinky_dip"]["coordinates"]},
                    "tip": {"name_vi": "Đầu ngón", "coordinates": data["right_hand"]["detailed"]["pinky_tip"]["coordinates"]}
                }
            }
        }
    
    # Face landmarks
    if results.face_landmarks:
        data["face"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                       for lm in results.face_landmarks.landmark]
    
    return data

def create_visualization_video():
    """Tạo video visualization với landmarks"""
    
    # Khởi tạo MediaPipe Holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Mở video input
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {input_video}")
        return
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Thông tin video:")
    print(f"   - FPS: {fps}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - Total frames: {total_frames}")
    
    # Khởi tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Danh sách lưu dữ liệu JSON
    frames_data = []
    frame_count = 0
    
    print("🔄 Đang xử lý và tạo video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Chuyển BGR sang RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý với MediaPipe
        results = holistic.process(rgb_frame)
        
        # Trích xuất landmarks chi tiết
        landmarks_data = extract_detailed_landmarks(results)
        
        # Vẽ landmarks lên frame
        annotated_frame = draw_detailed_landmarks(frame, results)
        
        # Thêm thông tin overlay
        final_frame = add_info_overlay(annotated_frame, frame_count, total_frames, fps, landmarks_data)
        
        # Ghi frame vào video output
        out.write(final_frame)
        
        # Lưu dữ liệu JSON
        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / fps,
            "landmarks": landmarks_data
        }
        frames_data.append(frame_data)
        
        # Hiển thị tiến trình
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Đã xử lý: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Đóng các tài nguyên
    cap.release()
    out.release()
    holistic.close()
    
    # Lưu dữ liệu JSON
    print(f"\n💾 Đang lưu dữ liệu JSON...")
    video_data = {
        "video_info": {
            "input_path": input_video,
            "output_path": output_video,
            "fps": fps,
            "resolution": [width, height],
            "total_frames": total_frames,
            "duration": total_frames / fps
        },
        "frames": frames_data
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Hoàn thành!")
    print(f"🎥 Video với landmarks: {output_video}")
    print(f"📄 Dữ liệu JSON: {output_json}")
    print(f"📊 Đã xử lý {frame_count} frames")
    
    # Thống kê
    frames_with_pose = sum(1 for frame in frames_data if frame['landmarks']['pose']['landmarks'])
    frames_with_left_hand = sum(1 for frame in frames_data if frame['landmarks']['left_hand']['landmarks'])
    frames_with_right_hand = sum(1 for frame in frames_data if frame['landmarks']['right_hand']['landmarks'])
    frames_with_face = sum(1 for frame in frames_data if frame['landmarks']['face'])
    
    print(f"\n📈 Thống kê detection:")
    print(f"   - Pose: {frames_with_pose}/{frame_count} frames ({frames_with_pose/frame_count*100:.1f}%)")
    print(f"   - Left hand: {frames_with_left_hand}/{frame_count} frames ({frames_with_left_hand/frame_count*100:.1f}%)")
    print(f"   - Right hand: {frames_with_right_hand}/{frame_count} frames ({frames_with_right_hand/frame_count*100:.1f}%)")
    print(f"   - Face: {frames_with_face}/{frame_count} frames ({frames_with_face/frame_count*100:.1f}%)")

if __name__ == "__main__":
    create_visualization_video()
