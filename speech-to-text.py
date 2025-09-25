import whisper
import pandas as pd
import re
from typing import List, Dict

def load_vocabulary_from_csv(csv_path: str) -> tuple:
    """
    Đọc file label.csv và trả về danh sách từ vựng + từ điển đồng nghĩa
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Tạo từ điển: từ gốc -> danh sách từ đồng nghĩa
        synonym_dict = {}
        # Tạo từ điển ngược: từ đồng nghĩa -> từ gốc
        reverse_synonym_dict = {}
        
        for _, row in df.iterrows():
            label = row['LABEL']
            synonyms_str = row.get('SYNONYMS', '')
            
            if pd.notna(synonyms_str) and synonyms_str.strip():
                # Tách các từ đồng nghĩa (cách nhau bởi dấu phẩy)
                synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                synonym_dict[label] = synonyms
                
                # Tạo mapping ngược
                for synonym in synonyms:
                    if synonym not in reverse_synonym_dict:
                        reverse_synonym_dict[synonym] = []
                    reverse_synonym_dict[synonym].append(label)
        
        # Lấy danh sách từ vựng gốc
        vocabulary = df['LABEL'].dropna().unique().tolist()
        
        print(f"✅ Đã tải {len(vocabulary)} từ vựng từ file CSV")
        print(f"📚 Có {len(synonym_dict)} từ vựng có từ đồng nghĩa")
        print(f"🔄 Tổng {sum(len(syns) for syns in synonym_dict.values())} từ đồng nghĩa")
        
        return vocabulary, synonym_dict, reverse_synonym_dict
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        return [], {}, {}

def is_valid_match(vocab_word: str, segment_text: str) -> bool:
    """
    Kiểm tra xem từ vựng có phù hợp với ngữ cảnh không
    """
    vocab_lower = vocab_word.lower().strip()
    text_lower = segment_text.lower()
    
    # Tìm vị trí của từ vựng trong text
    pattern = r'\b' + re.escape(vocab_lower) + r'\b'
    match = re.search(pattern, text_lower)
    
    if not match:
        return False
    
    start_pos = match.start()
    end_pos = match.end()
    
    # Kiểm tra ký tự trước và sau từ vựng
    char_before = text_lower[start_pos - 1] if start_pos > 0 else ' '
    char_after = text_lower[end_pos] if end_pos < len(text_lower) else ' '
    
    # Loại bỏ nếu từ vựng là một phần của từ ghép
    # Ví dụ: "hấp" trong "hấp dẫn" sẽ bị loại
    if char_after.isalpha() and not char_after.isspace():
        return False
    
    # Kiểm tra một số trường hợp đặc biệt
    common_compounds = {
        'hấp': ['hấp dẫn', 'hấp thụ'],
        'dẫn': ['hấp dẫn', 'dẫn đầu', 'dẫn dắt'],
        'tham': ['tham gia', 'tham khảo', 'tham quan'],
        'gia': ['tham gia', 'gia đình', 'gia tăng'],
        'khảo': ['tham khảo', 'khảo sát'],
        'quan': ['tham quan', 'quan tâm', 'quan trọng'],
        'tâm': ['quan tâm', 'tâm lý', 'trung tâm'],
        'trọng': ['quan trọng', 'trọng lượng'],
        'lý': ['tâm lý', 'lý do', 'xử lý'],
        'xử': ['xử lý', 'xử phạt'],
        'phạt': ['xử phạt', 'phạt tiền']
    }
    
    if vocab_lower in common_compounds:
        # Kiểm tra xem có phải là một phần của từ ghép không
        for compound in common_compounds[vocab_lower]:
            if compound in text_lower and compound != vocab_lower:
                return False
    
    return True

def find_vocabulary_in_text(transcription_result: Dict, vocabulary_list: List[str], synonym_dict: Dict, reverse_synonym_dict: Dict) -> List[Dict]:
    """
    Tìm các từ vựng từ danh sách trong text đã transcribe với kiểm tra ngữ cảnh và từ đồng nghĩa
    """
    found_words = []
    segments = transcription_result.get('segments', [])
    
    # Sắp xếp từ vựng theo độ dài giảm dần để ưu tiên từ ghép
    vocabulary_sorted = sorted(vocabulary_list, key=len, reverse=True)
    
    for segment in segments:
        segment_text = segment['text'].strip()
        segment_start = segment['start']
        segment_end = segment['end']
        text_lower = segment_text.lower()
        
        # Theo dõi các vị trí đã được match để tránh trùng lặp
        matched_positions = []
        
        # 1. Tìm từ vựng gốc trực tiếp
        for vocab_word in vocabulary_sorted:
            vocab_lower = vocab_word.lower().strip()
            
            pattern = r'\b' + re.escape(vocab_lower) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                
                position_taken = any(
                    start_pos < existing_end and end_pos > existing_start
                    for existing_start, existing_end in matched_positions
                )
                
                if not position_taken and is_valid_match(vocab_word, segment_text):
                    found_words.append({
                        'word': vocab_word,
                        'matched_text': vocab_word,
                        'match_type': 'direct',
                        'text_segment': segment_text,
                        'start_time': segment_start,
                        'end_time': segment_end,
                        'duration': segment_end - segment_start,
                        'confidence': len(vocab_word),
                        'position_in_text': start_pos
                    })
                    matched_positions.append((start_pos, end_pos))
        
        # 2. Tìm từ đồng nghĩa và suy ra từ vựng gốc
        for synonym, original_words in reverse_synonym_dict.items():
            synonym_lower = synonym.lower().strip()
            
            pattern = r'\b' + re.escape(synonym_lower) + r'\b'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                
                position_taken = any(
                    start_pos < existing_end and end_pos > existing_start
                    for existing_start, existing_end in matched_positions
                )
                
                if not position_taken and is_valid_match(synonym, segment_text):
                    # Thêm tất cả từ vựng gốc có từ đồng nghĩa này
                    for original_word in original_words:
                        found_words.append({
                            'word': original_word,
                            'matched_text': synonym,
                            'match_type': 'synonym',
                            'text_segment': segment_text,
                            'start_time': segment_start,
                            'end_time': segment_end,
                            'duration': segment_end - segment_start,
                            'confidence': len(synonym),
                            'position_in_text': start_pos
                        })
                    matched_positions.append((start_pos, end_pos))
    
    # Loại bỏ trùng lặp và sắp xếp theo độ tin cậy
    unique_words = []
    seen = set()
    for word_info in found_words:
        key = (word_info['word'], round(word_info['start_time'], 1))
        if key not in seen:
            seen.add(key)
            unique_words.append(word_info)
    
    # Sắp xếp theo thời gian, sau đó theo vị trí trong câu
    unique_words.sort(key=lambda x: (x['start_time'], x['position_in_text']))
    
    return unique_words

def format_time(seconds: float) -> str:
    """
    Chuyển đổi giây thành định dạng mm:ss
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def highlight_words_in_sentence(text: str, words: List[str]) -> str:
    """
    Tô sáng các từ vựng trong câu
    """
    highlighted_text = text
    # Sắp xếp từ theo độ dài giảm dần để tránh conflict
    words_sorted = sorted(words, key=len, reverse=True)
    
    for word in words_sorted:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        highlighted_text = re.sub(
            pattern, 
            f"**{word}**", 
            highlighted_text, 
            flags=re.IGNORECASE
        )
    
    return highlighted_text

def display_vocabulary_results(found_vocabulary: List[Dict], total_vocabulary: int):
    """
    Hiển thị kết quả tìm kiếm từ vựng - từng từ riêng biệt theo đúng thứ tự với hỗ trợ từ đồng nghĩa
    """
    print("\n" + "="*80)
    print("🎯 TỪ VỰNG TỪ LABEL.CSV XUẤT HIỆN TRONG VIDEO")
    print("="*80)
    
    if not found_vocabulary:
        print("❌ Không tìm thấy từ vựng nào từ danh sách label trong video")
        return
    
    # Đếm số từ vựng unique (không tính trùng lặp)
    unique_words = set(word_info['word'] for word_info in found_vocabulary)
    direct_matches = [w for w in found_vocabulary if w['match_type'] == 'direct']
    synonym_matches = [w for w in found_vocabulary if w['match_type'] == 'synonym']
    
    print(f"📊 Tìm thấy {len(unique_words)} từ vựng unique trong tổng số {total_vocabulary} từ vựng")
    print(f"📈 Tỷ lệ: {len(unique_words)/total_vocabulary*100:.1f}%")
    print(f"🎯 Trực tiếp: {len(direct_matches)} | 🔄 Qua đồng nghĩa: {len(synonym_matches)}")
    print("\n📝 CHI TIẾT CÁC TỪ VỰNG TÌM THẤY (theo thứ tự xuất hiện):")
    print("-"*80)
    
    # Sắp xếp tất cả từ vựng theo thời gian và vị trí trong câu
    found_vocabulary.sort(key=lambda x: (x['start_time'], x['position_in_text']))
    
    for i, word_info in enumerate(found_vocabulary, 1):
        start_time = format_time(word_info['start_time'])
        end_time = format_time(word_info['end_time'])
        
        # Tô sáng từ được match trong ngữ cảnh
        highlighted_text = highlight_words_in_sentence(word_info['text_segment'], [word_info['matched_text']])
        
        # Icon và màu sắc theo loại match
        match_icon = "🎯" if word_info['match_type'] == 'direct' else "🔄"
        
        print(f"{i:2d}. {match_icon} 🕐 {start_time}-{end_time}")
        print(f"    📌 Từ vựng: '{word_info['word']}'")
        
        if word_info['match_type'] == 'synonym':
            print(f"    🔄 Tìm qua từ đồng nghĩa: '{word_info['matched_text']}'")
        
        print(f"    💬 Ngữ cảnh: \"{highlighted_text}\"")
        print(f"    📍 Vị trí trong câu: ký tự thứ {word_info['position_in_text'] + 1}")
        print(f"    ⏱️  Thời lượng: {word_info['duration']:.1f}s")
        print()

# Tải model Whisper
print("🤖 Đang tải model Whisper...")
model = whisper.load_model("small")

# Transcribe video với timestamps
print("🎬 Đang xử lý video...")
result = model.transcribe("test.mp4", language="vi", word_timestamps=True, verbose=False)

# Hiển thị text đầy đủ
print("📜 Văn bản đầy đủ từ video:")
print(result["text"])

# Tải danh sách từ vựng từ label.csv
print("\n🔍 Đang tải danh sách từ vựng từ label.csv...")
vocabulary_list, synonym_dict, reverse_synonym_dict = load_vocabulary_from_csv("Dataset/Texts/label.csv")

if vocabulary_list:
    # Tìm từ vựng trong text
    print("🔎 Đang tìm kiếm từ vựng trong text video...")
    found_vocabulary = find_vocabulary_in_text(result, vocabulary_list, synonym_dict, reverse_synonym_dict)
    
    # Hiển thị kết quả
    display_vocabulary_results(found_vocabulary, len(vocabulary_list))

print("\n✅ Hoàn thành!")
