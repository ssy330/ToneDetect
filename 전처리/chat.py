
from initial import replace_initial_abbr
from emotion import preprocess_text
from emoticon import remove_emoticons
from initial import replace_initial_abbr


# pipeline 에서 호출할 run_chat 정의
def run_chat(msgs):
    for msg in msgs:
        text_for_chat = msg["cleaned_text"]

        # (2) 오로지 초성 치환만 수행
        expanded = replace_initial_abbr(text_for_chat)
        msg["cleaned_text"] = expanded
        
        #result = full_preprocess(text)
        #msg["text"] = result["step3_no_emoticons"]
        #msg["emotion_chunks"] = result["emotion_chunks"]
        #msg["extracted_emoticons"] = result["extracted_emoticons"]
    return msgs

def full_preprocess(text: str) -> dict:
    original_text = text  # ✅ 이 줄을 먼저!

    # 1️⃣ 초성 축약어 치환
    step1_abbr_expanded = replace_initial_abbr(text)

    # 2️⃣ 감정 블록 처리
    emotion_result = preprocess_text(step1_abbr_expanded)
    
    # ✅ ✅ 핵심 변경! — 이모지 추출은 원본 또는 step1_abbr_expanded 기준으로!
    #emoticons = extract_emoticons(step1_abbr_expanded)  # 여기! text = step1_abbr_expanded 상태

    # 3️⃣ 이모티콘 추출 및 제거
    #emoticons = extract_emoticons(emotion_result["text_input"])
    cleaned_text = remove_emoticons(emotion_result["text_input"])

    return {
        "original": original_text,                  # ✅ 진짜 원문 그대로
        "step1_abbr_expanded": step1_abbr_expanded,
        "step2_no_emotion": emotion_result["text_input"],
        "step3_no_emoticons": cleaned_text,
        "emotion_chunks": emotion_result["emotion_chunks"],
        # "extracted_emoticons": emoticons
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