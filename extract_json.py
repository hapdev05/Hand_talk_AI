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
    """Trích xuất tất cả landmarks từ holistic (pose + hands + face) đồng bộ với Mixamo bone structure"""
    
    # Cấu trúc hierarchy xương Mixamo chuẩn
    mixamo_hierarchy = {
        "Hips": {  # Root bone
            "children": ["Spine", "LeftUpLeg", "RightUpLeg"]
        },
        "Spine": {
            "children": ["Spine1", "Spine2", "Neck", "LeftShoulder", "RightShoulder"]
        },
        "Neck": {
            "children": ["Head"]
        },
        "Head": {
            "children": ["LeftEye", "RightEye"]
        },
        "LeftShoulder": {
            "children": ["LeftArm"]
        },
        "LeftArm": {
            "children": ["LeftForeArm"]
        },
        "LeftForeArm": {
            "children": ["LeftHand"]
        },
        "LeftHand": {
            "children": ["LeftHandThumb1", "LeftHandIndex1", "LeftHandMiddle1", "LeftHandRing1", "LeftHandPinky1"]
        },
        "RightShoulder": {
            "children": ["RightArm"]
        },
        "RightArm": {
            "children": ["RightForeArm"]
        },
        "RightForeArm": {
            "children": ["RightHand"]
        },
        "RightHand": {
            "children": ["RightHandThumb1", "RightHandIndex1", "RightHandMiddle1", "RightHandRing1", "RightHandPinky1"]
        },
        "LeftUpLeg": {
            "children": ["LeftLeg"]
        },
        "LeftLeg": {
            "children": ["LeftFoot"]
        },
        "LeftFoot": {
            "children": ["LeftToeBase"]
        },
        "RightUpLeg": {
            "children": ["RightLeg"]
        },
        "RightLeg": {
            "children": ["RightFoot"]
        },
        "RightFoot": {
            "children": ["RightToeBase"]
        }
    }
    
    # Mapping tên các landmarks cho tay theo chuẩn Mixamo (21 điểm cho mỗi tay)
    hand_landmark_names = [
        "Hand",            # 0  - Cổ tay -> Hand (wrist)
        "HandThumb1",      # 1  - Thumb Carpometacarpal -> HandThumb1
        "HandThumb2",      # 2  - Thumb Metacarpophalangeal -> HandThumb2  
        "HandThumb3",      # 3  - Thumb Interphalangeal -> HandThumb3
        "HandThumb4",      # 4  - Đầu ngón cái -> HandThumb4 (tip)
        "HandIndex1",      # 5  - Index Metacarpophalangeal -> HandIndex1
        "HandIndex2",      # 6  - Index Proximal Interphalangeal -> HandIndex2
        "HandIndex3",      # 7  - Index Distal Interphalangeal -> HandIndex3
        "HandIndex4",      # 8  - Đầu ngón trỏ -> HandIndex4 (tip)
        "HandMiddle1",     # 9  - Middle Metacarpophalangeal -> HandMiddle1
        "HandMiddle2",     # 10 - Middle Proximal Interphalangeal -> HandMiddle2
        "HandMiddle3",     # 11 - Middle Distal Interphalangeal -> HandMiddle3
        "HandMiddle4",     # 12 - Đầu ngón giữa -> HandMiddle4 (tip)
        "HandRing1",       # 13 - Ring Metacarpophalangeal -> HandRing1
        "HandRing2",       # 14 - Ring Proximal Interphalangeal -> HandRing2
        "HandRing3",       # 15 - Ring Distal Interphalangeal -> HandRing3
        "HandRing4",       # 16 - Đầu ngón áp út -> HandRing4 (tip)
        "HandPinky1",      # 17 - Pinky Metacarpophalangeal -> HandPinky1
        "HandPinky2",      # 18 - Pinky Proximal Interphalangeal -> HandPinky2
        "HandPinky3",      # 19 - Pinky Distal Interphalangeal -> HandPinky3
        "HandPinky4"       # 20 - Đầu ngón út -> HandPinky4 (tip)
    ]
    
    # Mapping tên các landmarks cho pose theo chuẩn Mixamo (33 điểm chính)
    pose_landmark_names = [
        "Head",                    # 0  - nose -> Head
        "LeftEye",                 # 1  - left_eye_inner -> LeftEye  
        "LeftEye",                 # 2  - left_eye -> LeftEye
        "LeftEye",                 # 3  - left_eye_outer -> LeftEye
        "RightEye",                # 4  - right_eye_inner -> RightEye
        "RightEye",                # 5  - right_eye -> RightEye
        "RightEye",                # 6  - right_eye_outer -> RightEye
        "Head",                    # 7  - left_ear -> Head
        "Head",                    # 8  - right_ear -> Head
        "Head",                    # 9  - mouth_left -> Head
        "Head",                    # 10 - mouth_right -> Head
        "LeftShoulder",            # 11 - left_shoulder -> LeftShoulder
        "RightShoulder",           # 12 - right_shoulder -> RightShoulder
        "LeftArm",                 # 13 - left_elbow -> LeftArm (elbow)
        "RightArm",                # 14 - right_elbow -> RightArm (elbow)
        "LeftForeArm",             # 15 - left_wrist -> LeftForeArm (wrist)
        "RightForeArm",            # 16 - right_wrist -> RightForeArm (wrist)
        "LeftHand",                # 17 - left_pinky -> LeftHand
        "RightHand",               # 18 - right_pinky -> RightHand
        "LeftHand",                # 19 - left_index -> LeftHand
        "RightHand",               # 20 - right_index -> RightHand
        "LeftHand",                # 21 - left_thumb -> LeftHand
        "RightHand",               # 22 - right_thumb -> RightHand
        "LeftUpLeg",               # 23 - left_hip -> LeftUpLeg (hip)
        "RightUpLeg",              # 24 - right_hip -> RightUpLeg (hip)
        "LeftLeg",                 # 25 - left_knee -> LeftLeg (knee)
        "RightLeg",                # 26 - right_knee -> RightLeg (knee)
        "LeftFoot",                # 27 - left_ankle -> LeftFoot (ankle)
        "RightFoot",               # 28 - right_ankle -> RightFoot (ankle)
        "LeftFoot",                # 29 - left_heel -> LeftFoot
        "RightFoot",               # 30 - right_heel -> RightFoot
        "LeftToeBase",             # 31 - left_foot_index -> LeftToeBase
        "RightToeBase"             # 32 - right_foot_index -> RightToeBase
    ]
    
    data = {
        "mixamo_hierarchy": mixamo_hierarchy,  # Cấu trúc xương Mixamo
        "pose": {
            "landmarks": [],
            "detailed": {},
            "mixamo_bones": {}  # Mapping theo tên xương Mixamo
        },
        "left_hand": {
            "landmarks": [],
            "detailed": {},
            "fingers": {},
            "mixamo_bones": {}  # Mapping theo tên xương Mixamo
        },
        "right_hand": {
            "landmarks": [],
            "detailed": {},
            "fingers": {},
            "mixamo_bones": {}  # Mapping theo tên xương Mixamo
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
            mixamo_bone_name = pose_landmark_names[i] if i < len(pose_landmark_names) else f"UnknownBone_{i}"
            
            data["pose"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)],
                "visibility": float(landmark.visibility),
                "mixamo_bone": mixamo_bone_name
            }
            
            # Mapping theo tên xương Mixamo
            if mixamo_bone_name not in data["pose"]["mixamo_bones"]:
                data["pose"]["mixamo_bones"][mixamo_bone_name] = []
            data["pose"]["mixamo_bones"][mixamo_bone_name].append({
                "mediapipe_index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)],
                "visibility": float(landmark.visibility)
            })
    
    # Left hand landmarks với chi tiết ngón tay
    if results.left_hand_landmarks:
        # Raw landmarks
        data["left_hand"]["landmarks"] = [[float(lm.x), float(lm.y), float(lm.z)] 
                                        for lm in results.left_hand_landmarks.landmark]
        
        # Chi tiết từng landmark
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            landmark_name = hand_landmark_names[i] if i < len(hand_landmark_names) else f"hand_landmark_{i}"
            mixamo_bone_name = f"Left{hand_landmark_names[i]}" if i < len(hand_landmark_names) else f"LeftUnknownBone_{i}"
            
            data["left_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)],
                "mixamo_bone": mixamo_bone_name
            }
            
            # Mapping theo tên xương Mixamo
            data["left_hand"]["mixamo_bones"][mixamo_bone_name] = {
                "mediapipe_index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay với tên Mixamo
        data["left_hand"]["fingers"] = {
            "thumb": {  # Ngón cái
                "name_vi": "Ngón cái",
                "mixamo_bone_prefix": "LeftHandThumb",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "LeftHandThumb1",
                        "coordinates": data["left_hand"]["detailed"]["HandThumb1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "LeftHandThumb2",
                        "coordinates": data["left_hand"]["detailed"]["HandThumb2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "LeftHandThumb3",
                        "coordinates": data["left_hand"]["detailed"]["HandThumb3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "LeftHandThumb4",
                        "coordinates": data["left_hand"]["detailed"]["HandThumb4"]["coordinates"]
                    }
                }
            },
            "index": {  # Ngón trỏ
                "name_vi": "Ngón trỏ",
                "mixamo_bone_prefix": "LeftHandIndex",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "LeftHandIndex1",
                        "coordinates": data["left_hand"]["detailed"]["HandIndex1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "LeftHandIndex2",
                        "coordinates": data["left_hand"]["detailed"]["HandIndex2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "LeftHandIndex3",
                        "coordinates": data["left_hand"]["detailed"]["HandIndex3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "LeftHandIndex4",
                        "coordinates": data["left_hand"]["detailed"]["HandIndex4"]["coordinates"]
                    }
                }
            },
            "middle": {  # Ngón giữa
                "name_vi": "Ngón giữa",
                "mixamo_bone_prefix": "LeftHandMiddle",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "LeftHandMiddle1",
                        "coordinates": data["left_hand"]["detailed"]["HandMiddle1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "LeftHandMiddle2",
                        "coordinates": data["left_hand"]["detailed"]["HandMiddle2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "LeftHandMiddle3",
                        "coordinates": data["left_hand"]["detailed"]["HandMiddle3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "LeftHandMiddle4",
                        "coordinates": data["left_hand"]["detailed"]["HandMiddle4"]["coordinates"]
                    }
                }
            },
            "ring": {  # Ngón áp út
                "name_vi": "Ngón áp út",
                "mixamo_bone_prefix": "LeftHandRing",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "LeftHandRing1",
                        "coordinates": data["left_hand"]["detailed"]["HandRing1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "LeftHandRing2",
                        "coordinates": data["left_hand"]["detailed"]["HandRing2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "LeftHandRing3",
                        "coordinates": data["left_hand"]["detailed"]["HandRing3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "LeftHandRing4",
                        "coordinates": data["left_hand"]["detailed"]["HandRing4"]["coordinates"]
                    }
                }
            },
            "pinky": {  # Ngón út
                "name_vi": "Ngón út",
                "mixamo_bone_prefix": "LeftHandPinky",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "LeftHandPinky1",
                        "coordinates": data["left_hand"]["detailed"]["HandPinky1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "LeftHandPinky2",
                        "coordinates": data["left_hand"]["detailed"]["HandPinky2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "LeftHandPinky3",
                        "coordinates": data["left_hand"]["detailed"]["HandPinky3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "LeftHandPinky4",
                        "coordinates": data["left_hand"]["detailed"]["HandPinky4"]["coordinates"]
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
            mixamo_bone_name = f"Right{hand_landmark_names[i]}" if i < len(hand_landmark_names) else f"RightUnknownBone_{i}"
            
            data["right_hand"]["detailed"][landmark_name] = {
                "index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)],
                "mixamo_bone": mixamo_bone_name
            }
            
            # Mapping theo tên xương Mixamo
            data["right_hand"]["mixamo_bones"][mixamo_bone_name] = {
                "mediapipe_index": i,
                "coordinates": [float(landmark.x), float(landmark.y), float(landmark.z)]
            }
        
        # Nhóm theo ngón tay với tên Mixamo
        data["right_hand"]["fingers"] = {
            "thumb": {
                "name_vi": "Ngón cái",
                "mixamo_bone_prefix": "RightHandThumb",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "RightHandThumb1",
                        "coordinates": data["right_hand"]["detailed"]["HandThumb1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "RightHandThumb2",
                        "coordinates": data["right_hand"]["detailed"]["HandThumb2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "RightHandThumb3",
                        "coordinates": data["right_hand"]["detailed"]["HandThumb3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "RightHandThumb4",
                        "coordinates": data["right_hand"]["detailed"]["HandThumb4"]["coordinates"]
                    }
                }
            },
            "index": {
                "name_vi": "Ngón trỏ",
                "mixamo_bone_prefix": "RightHandIndex",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "RightHandIndex1",
                        "coordinates": data["right_hand"]["detailed"]["HandIndex1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "RightHandIndex2",
                        "coordinates": data["right_hand"]["detailed"]["HandIndex2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "RightHandIndex3",
                        "coordinates": data["right_hand"]["detailed"]["HandIndex3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "RightHandIndex4",
                        "coordinates": data["right_hand"]["detailed"]["HandIndex4"]["coordinates"]
                    }
                }
            },
            "middle": {
                "name_vi": "Ngón giữa",
                "mixamo_bone_prefix": "RightHandMiddle",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "RightHandMiddle1",
                        "coordinates": data["right_hand"]["detailed"]["HandMiddle1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "RightHandMiddle2",
                        "coordinates": data["right_hand"]["detailed"]["HandMiddle2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "RightHandMiddle3",
                        "coordinates": data["right_hand"]["detailed"]["HandMiddle3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "RightHandMiddle4",
                        "coordinates": data["right_hand"]["detailed"]["HandMiddle4"]["coordinates"]
                    }
                }
            },
            "ring": {
                "name_vi": "Ngón áp út",
                "mixamo_bone_prefix": "RightHandRing",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "RightHandRing1",
                        "coordinates": data["right_hand"]["detailed"]["HandRing1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "RightHandRing2",
                        "coordinates": data["right_hand"]["detailed"]["HandRing2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "RightHandRing3",
                        "coordinates": data["right_hand"]["detailed"]["HandRing3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "RightHandRing4",
                        "coordinates": data["right_hand"]["detailed"]["HandRing4"]["coordinates"]
                    }
                }
            },
            "pinky": {
                "name_vi": "Ngón út",
                "mixamo_bone_prefix": "RightHandPinky",
                "joints": {
                    "joint1": {
                        "name_vi": "Khớp gốc",
                        "mixamo_bone": "RightHandPinky1",
                        "coordinates": data["right_hand"]["detailed"]["HandPinky1"]["coordinates"]
                    },
                    "joint2": {
                        "name_vi": "Khớp giữa",
                        "mixamo_bone": "RightHandPinky2",
                        "coordinates": data["right_hand"]["detailed"]["HandPinky2"]["coordinates"]
                    },
                    "joint3": {
                        "name_vi": "Khớp đầu",
                        "mixamo_bone": "RightHandPinky3",
                        "coordinates": data["right_hand"]["detailed"]["HandPinky3"]["coordinates"]
                    },
                    "tip": {
                        "name_vi": "Đầu ngón",
                        "mixamo_bone": "RightHandPinky4",
                        "coordinates": data["right_hand"]["detailed"]["HandPinky4"]["coordinates"]
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