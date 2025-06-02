import re
from collections import defaultdict

# 감정 문자와 태그 정의
EMOTION_CHARS = set("ㅋㅎㅠㅜ?!~.;")
TAG_RULES = {
    'ㅋ': '[웃김]',
    'ㅎ': '[웃김]',
    'ㅠ': '[슬픔]',
    'ㅜ': '[슬픔]',
    '?': '[의문]',
    '!': '[강조]',
    '~': '[여운]',
    '.': '[망설임]',
    ';': '[난처함]'
}

# 1. 큐/쿠는 감정 문맥에서 ㅋㅠ, ㅋㅜ로 치환
def normalize_qu_patterns(text):
    text = re.sub(r'(?<=[ㅋㅎ])큐(?=[ㅠㅜㅋㅎ])', 'ㅋㅠ', text)
    text = re.sub(r'(?<=[ㅋㅎ])쿠(?=[ㅠㅜㅋㅎ])', 'ㅋㅜ', text)
    return text

# 2. 감정 블록 추출 (길이 1도 포함하되, 단독 ? 또는 . 제외)
def extract_emotion_chunks(text):
    pattern = f"[{''.join(re.escape(ch) for ch in EMOTION_CHARS)}ㄱㄲㅌ]+"
    chunks = re.findall(pattern, text)

    filtered = []
    for chunk in chunks:
        cleaned = ''.join(ch for ch in chunk if ch in EMOTION_CHARS)
        # 단독 ? 또는 .는 제외
        if len(cleaned) == 1 and cleaned in '.?':
            continue
        filtered.append(chunk)
    
    return filtered

# 3. 감정 블록 분류 (ㄱㄲㅌ 제거하고 태그 추출)
def classify_emotion_chunks(chunks):
    tag_dict = defaultdict(list)
    for chunk in chunks:
        cleaned = ''.join(ch for ch in chunk if ch in EMOTION_CHARS)
        unique_chars = set(cleaned)
        tags = sorted(set(TAG_RULES[ch] for ch in unique_chars))
        tag_key = ''.join(tags)
        tag_dict[tag_key].append(chunk)
    return dict(tag_dict)

# 4. 감정 블록 제거하되, ? 또는 . 포함 시 해당 기호로 대체
def remove_emotion_chunks_preserve_punct(text):
    def replace_emotion_block(m):
        block = m.group(0)
        if '?' in block:
            return '? '
        elif '.' in block:
            return '. '
        else:
            return ' '
    pattern = f"[{''.join(re.escape(ch) for ch in EMOTION_CHARS)}ㄱㄲㅌ]+"
    cleaned = re.sub(pattern, replace_emotion_block, text)
    return re.sub(r'\s+', ' ', cleaned).strip()

# 5. 전체 전처리 함수
def preprocess_text(text):
    original_text = text
    text = normalize_qu_patterns(text)
    emotion_chunks = extract_emotion_chunks(text)
    emotion_classified = classify_emotion_chunks(emotion_chunks)
    text_input = remove_emotion_chunks_preserve_punct(text)
    return {
        "original": original_text,
        "text_input": text_input,
        "emotion_chunks": emotion_classified
    }

# 테스트
if __name__ == "__main__":
    text = "이게 맞아?ㅋㅋ큐ㅠㅠ진심으로 궁금해요. 안녕하세요? ...!"
    res = preprocess_text(text)

    print("🟡 원문:", res["original"])
    print("🟢 정제:", res["text_input"])
    print("🔵 감정 블록:", res["emotion_chunks"])
