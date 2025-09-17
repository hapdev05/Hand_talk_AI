import cv2
import mediapipe as mp
import json
import numpy as np

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Cấu hình đường dẫn
video_path = "xang.mp4"  # Đường dẫn video đầu vào
output_json = "pose_analysis.json"  # File JSON đầu ra

print("🎬 Bắt đầu phân tích pose từ video...")
print(f"📹 Video: {video_path}")
print(f"📄 Output: {output_json}")

# ====================
# Các hàm xử lý
# ====================
def extract_pose_landmarks(results):
    """Trích xuất pose landmarks từ kết quả MediaPipe"""
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([
                float(landmark.x),
                float(landmark.y), 
                float(landmark.z),
                float(landmark.visibility)
            ])
    return landmarks

def extract_holistic_landmarks(results):
    """Trích xuất tất cả landmarks từ holistic (pose + hands + face) với chi tiết ngón tay"""
    
    # Mapping tên các landmarks cho tay (21 điểm cho mỗi tay)
    hand_landmark_names = [
        "wrist",           # 0  - Cổ tay
        "thumb_cmc",       # 1  - Thumb Carpometacarpal (khớp gốc ngón cái)
        "thumb_mcp",       # 2  - Thumb Metacarpophalangeal (khớp giữa ngón cái)
        "thumb_ip",        # 3  - Thumb Interphalangeal (khớp ngón cái)
        "thumb_tip",       # 4  - Đầu ngón cái
        "index_mcp",       # 5  - Index Metacarpophalangeal (khớp gốc ngón trỏ)
        "index_pip",       # 6  - Index Proximal Interphalangeal (khớp giữa ngón trỏ)
        "index_dip",       # 7  - Index Distal Interphalangeal (khớp đầu ngón trỏ)
        "index_tip",       # 8  - Đầu ngón trỏ
        "middle_mcp",      # 9  - Middle Metacarpophalangeal (khớp gốc ngón giữa)
        "middle_pip",      # 10 - Middle Proximal Interphalangeal (khớp giữa ngón giữa)
        "middle_dip",      # 11 - Middle Distal Interphalangeal (khớp đầu ngón giữa)
        "middle_tip",      # 12 - Đầu ngón giữa
        "ring_mcp",        # 13 - Ring Metacarpophalangeal (khớp gốc ngón áp út)
        "ring_pip",        # 14 - Ring Proximal Interphalangeal (khớp giữa ngón áp út)
        "ring_dip",        # 15 - Ring Distal Interphalangeal (khớp đầu ngón áp út)
        "ring_tip",        # 16 - Đầu ngón áp út
        "pinky_mcp",       # 17 - Pinky Metacarpophalangeal (khớp gốc ngón út)
        "pinky_pip",       # 18 - Pinky Proximal Interphalangeal (khớp giữa ngón út)
        "pinky_dip",       # 19 - Pinky Distal Interphalangeal (khớp đầu ngón út)
        "pinky_tip"        # 20 - Đầu ngón út
    ]
    
    # Mapping tên các landmarks cho pose (33 điểm chính)
    pose_landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    data = {
        "pose": {
            "landmarks": [],
            "detailed": {}
        },
        "left_hand": {
            "landmarks": [],
            "detailed": {},
            "fingers": {}
        },
        "right_hand": {
            "landmarks": [],
            "detailed": {},
            "fingers": {}
        },
        "face": []
    }
    
    # Pose landmarks với tên chi tiết
    if results.pose_landmarks:
        # Raw landmarks
        data["pose"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)] 
                                   for lm in results.pose_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = pose_landmark_names[i] if i < len(pose_landmark_names) else f"pose_landmark_{i}"
            data["pose"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)],
                "visibility": float(landmark.visibility)
            }
    
    # Left hand landmarks với chi tiết ngón tay
    if results.left_hand_landmarks:
        # Raw landmarks
        data["left_hand"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                                        for lm in results.left_hand_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            landmark_name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"hand_landmark_{i}"
            data["left_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay với tên tiếng Việt
        data["left_hand"]["fingers"] = {
            "thumb": {  # Ngón cái
                "name_vi": "Ngón cái",
                "joints": {
                    "cmc": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["left_hand"]["detailed"]["thumb_cmc"]["coordinates"]
                    },
                    "mcp": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["left_hand"]["detailed"]["thumb_mcp"]["coordinates"]
                    },
                    "ip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["left_hand"]["detailed"]["thumb_ip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["left_hand"]["detailed"]["thumb_tip"]["coordinates"]
                    }
                }
            },
            "index": {  # Ngón trỏ
                "name_vi": "Ngón trỏ",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["left_hand"]["detailed"]["index_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["left_hand"]["detailed"]["index_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["left_hand"]["detailed"]["index_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["left_hand"]["detailed"]["index_tip"]["coordinates"]
                    }
                }
            },
            "middle": {  # Ngón giữa
                "name_vi": "Ngón giữa",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["left_hand"]["detailed"]["middle_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["left_hand"]["detailed"]["middle_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["left_hand"]["detailed"]["middle_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["left_hand"]["detailed"]["middle_tip"]["coordinates"]
                    }
                }
            },
            "ring": {  # Ngón áp út
                "name_vi": "Ngón áp út",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["left_hand"]["detailed"]["ring_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["left_hand"]["detailed"]["ring_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["left_hand"]["detailed"]["ring_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["left_hand"]["detailed"]["ring_tip"]["coordinates"]
                    }
                }
            },
            "pinky": {  # Ngón út
                "name_vi": "Ngón út",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["left_hand"]["detailed"]["pinky_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["left_hand"]["detailed"]["pinky_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["left_hand"]["detailed"]["pinky_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["left_hand"]["detailed"]["pinky_tip"]["coordinates"]
                    }
                }
            }
        }
    
    # Right hand landmarks với chi tiết ngón tay (tương tự left hand)
    if results.right_hand_landmarks:
        # Raw landmarks
        data["right_hand"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                                         for lm in results.right_hand_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            landmark_name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"hand_landmark_{i}"
            data["right_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay với tên tiếng Việt
        data["right_hand"]["fingers"] = {
            "thumb": {
                "name_vi": "Ngón cái",
                "joints": {
                    "cmc": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["right_hand"]["detailed"]["thumb_cmc"]["coordinates"]
                    },
                    "mcp": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["right_hand"]["detailed"]["thumb_mcp"]["coordinates"]
                    },
                    "ip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["right_hand"]["detailed"]["thumb_ip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["right_hand"]["detailed"]["thumb_tip"]["coordinates"]
                    }
                }
            },
            "index": {
                "name_vi": "Ngón trỏ",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["right_hand"]["detailed"]["index_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["right_hand"]["detailed"]["index_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["right_hand"]["detailed"]["index_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["right_hand"]["detailed"]["index_tip"]["coordinates"]
                    }
                }
            },
            "middle": {
                "name_vi": "Ngón giữa",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["right_hand"]["detailed"]["middle_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["right_hand"]["detailed"]["middle_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["right_hand"]["detailed"]["middle_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["right_hand"]["detailed"]["middle_tip"]["coordinates"]
                    }
                }
            },
            "ring": {
                "name_vi": "Ngón áp út",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["right_hand"]["detailed"]["ring_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["right_hand"]["detailed"]["ring_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["right_hand"]["detailed"]["ring_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["right_hand"]["detailed"]["ring_tip"]["coordinates"]
                    }
                }
            },
            "pinky": {
                "name_vi": "Ngón út",
                "joints": {
                    "mcp": {
                        "name_vi": "Khớp gốc",
                        "coordinates": data["right_hand"]["detailed"]["pinky_mcp"]["coordinates"]
                    },
                    "pip": {
                        "name_vi": "Khớp giữa",
                        "coordinates": data["right_hand"]["detailed"]["pinky_pip"]["coordinates"]
                    },
                    "dip": {
                        "name_vi": "Khớp đầu",
                        "coordinates": data["right_hand"]["detailed"]["pinky_dip"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "coordinates": data["right_hand"]["detailed"]["pinky_tip"]["coordinates"]
                    }
                }
            }
        }
    
    # Face landmarks (468 điểm - giữ nguyên)
    if results.face_landmarks:
        data["face"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                       for lm in results.face_landmarks.landmark]
    
    return data

def analyze_video_pose(video_path, use_holistic=True):
    """Phân tích pose từ video và trả về dữ liệu landmarks"""
    
    # Khởi tạo model
    if use_holistic:
        model = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        model = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return None
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Thông tin video:")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Tổng frames: {total_frames}")
    print(f"   - Thời lượng: {duration:.2f}s")
    
    results_list = []
    frame_count = 0
    
    print("🔄 Đang xử lý video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Chuyển BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý frame
        results = model.process(rgb_frame)
        
        # Trích xuất landmarks
        if use_holistic:
            landmarks_data = extract_holistic_landmarks(results)
        else:
            landmarks_data = extract_pose_landmarks(results)
        
        # Lưu kết quả
        frame_data = {
            "frame": frame_count,
            "timestamp": frame_count / fps if fps > 0 else 0,
            "landmarks": landmarks_data
        }
        
        results_list.append(frame_data)
        
        # Hiển thị tiến trình
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Đã xử lý: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    model.close()
    
    print(f"✅ Hoàn thành! Đã xử lý {frame_count} frames")
    
    return {
        "video_info": {
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        },
        "frames": results_list
    }

# ====================
# Chương trình chính
# ====================

def main():
    try:
        # Phân tích video với holistic (pose + hands + face)
        print("\n🎯 Phân tích holistic (pose + hands + face)...")
        data = analyze_video_pose(video_path, use_holistic=True)
        
        if data is None:
            print("❌ Lỗi khi phân tích video!")
            return
        
        # Lưu kết quả ra JSON
        print(f"\n💾 Đang lưu kết quả vào {output_json}...")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Đã lưu thành công!")
        print(f"📄 File output: {output_json}")
        print(f"📊 Số frames đã phân tích: {len(data['frames'])}")
        
        # Hiển thị một số thống kê chi tiết
        frames_with_pose = sum(1 for frame in data['frames'] if frame['landmarks']['pose']['landmarks'])
        frames_with_left_hand = sum(1 for frame in data['frames'] if frame['landmarks']['left_hand']['landmarks'])
        frames_with_right_hand = sum(1 for frame in data['frames'] if frame['landmarks']['right_hand']['landmarks'])
        frames_with_face = sum(1 for frame in data['frames'] if frame['landmarks']['face'])
        
        print(f"\n📈 Thống kê chi tiết:")
        print(f"   - Frames có pose: {frames_with_pose}/{len(data['frames'])} ({frames_with_pose/len(data['frames'])*100:.1f}%)")
        print(f"   - Frames có tay trái: {frames_with_left_hand}/{len(data['frames'])} ({frames_with_left_hand/len(data['frames'])*100:.1f}%)")
        print(f"   - Frames có tay phải: {frames_with_right_hand}/{len(data['frames'])} ({frames_with_right_hand/len(data['frames'])*100:.1f}%)")
        print(f"   - Frames có khuôn mặt: {frames_with_face}/{len(data['frames'])} ({frames_with_face/len(data['frames'])*100:.1f}%)")
        
        # Thống kê chi tiết về ngón tay (từ frame đầu tiên có hand data)
        sample_frame = None
        for frame in data['frames']:
            if frame['landmarks']['left_hand']['landmarks'] or frame['landmarks']['right_hand']['landmarks']:
                sample_frame = frame
                break
        
        if sample_frame:
            print(f"\n🖐️ Chi tiết khớp ngón tay (mẫu từ frame {sample_frame['frame']}):")
            
            # Left hand
            if sample_frame['landmarks']['left_hand']['landmarks']:
                print(f"   👈 Tay trái:")
                print(f"      - Tổng landmarks: {len(sample_frame['landmarks']['left_hand']['landmarks'])}")
                fingers = sample_frame['landmarks']['left_hand']['fingers']
                for finger_name, finger_data in fingers.items():
                    joints_count = len(finger_data['joints'])
                    print(f"      - {finger_data['name_vi']}: {joints_count} khớp")
                    for joint_name, joint_data in finger_data['joints'].items():
                        x, y, z = joint_data['coordinates']
                        print(f"        * {joint_data['name_vi']}: ({x:.3f}, {y:.3f}, {z:.3f})")
            
            # Right hand
            if sample_frame['landmarks']['right_hand']['landmarks']:
                print(f"   👉 Tay phải:")
                print(f"      - Tổng landmarks: {len(sample_frame['landmarks']['right_hand']['landmarks'])}")
                fingers = sample_frame['landmarks']['right_hand']['fingers']
                for finger_name, finger_data in fingers.items():
                    joints_count = len(finger_data['joints'])
                    print(f"      - {finger_data['name_vi']}: {joints_count} khớp")
                    for joint_name, joint_data in finger_data['joints'].items():
                        x, y, z = joint_data['coordinates']
                        print(f"        * {joint_data['name_vi']}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Hiển thị thông tin về file đã lưu
        import os
        file_size = os.path.getsize(output_json) / (1024*1024)  # MB
        print(f"\n💾 Thông tin file JSON:")
        print(f"   - Kích thước: {file_size:.2f} MB")
        print(f"   - Đường dẫn: {os.path.abspath(output_json)}")
        
        print(f"\n📋 Cấu trúc dữ liệu JSON:")
        print(f"   - pose: 33 điểm (x, y, z, visibility) + tên chi tiết")
        print(f"   - left_hand/right_hand:")
        print(f"     * landmarks: 21 điểm raw (x, y, z)")
        print(f"     * detailed: từng landmark có tên và index")
        print(f"     * fingers: nhóm theo 5 ngón tay với tên tiếng Việt")
        print(f"       - Mỗi ngón có 3-4 khớp với tọa độ chi tiết")
        print(f"   - face: 468 điểm (x, y, z)")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()