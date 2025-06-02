# text_cleaner.py

from initial import replace_initial_abbr
from emotion import preprocess_text
from emoticon import extract_emoticons, remove_emoticons

def full_preprocess(text: str) -> dict:
    original_text = text  # ✅ 이 줄을 먼저!

    # 1️⃣ 초성 축약어 치환
    text = replace_initial_abbr(text)

    # 2️⃣ 감정 블록 처리
    emotion_result = preprocess_text(text)

    # 3️⃣ 이모티콘 추출 및 제거
    emoticons = extract_emoticons(emotion_result["text_input"])
    cleaned_text = remove_emoticons(emotion_result["text_input"])

    return {
        "original": original_text,                  # ✅ 진짜 원문 그대로
        "step1_abbr_expanded": text,
        "step2_no_emotion": emotion_result["text_input"],
        "step3_no_emoticons": cleaned_text,
        "emotion_chunks": emotion_result["emotion_chunks"],
        "extracted_emoticons": emoticons
    }

# 테스트 예시
if __name__ == "__main__":
    sample = "ㅇㅇ 이게 맞아?ㅋㅋ큐ㅠㅠ 진심으로 궁금해요 ㅇㅅㅇ >_< ㅡㅡ ❤️🔥🙂ㅋㅋㅋㅠㅠ"
    result = full_preprocess(sample)

    print("🟡 원문:", result["original"])
    print("① 초성 확장:", result["step1_abbr_expanded"])
    print("② 감정 제거:", result["step2_no_emotion"])
    print("③ 최종 정제:", result["step3_no_emoticons"])
    print("🔵 감정 블록:", result["emotion_chunks"])
    print("🟣 이모티콘:", result["extracted_emoticons"])