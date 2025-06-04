import json
from chat import full_preprocess

# pipeline 에서 호출할 run_clean 정의
def run_clean(msgs):
    for msg in msgs:
        # 1) 원문 보관
        text = msg["text"]  

        # 2) full_preprocess 실행
        result = full_preprocess(text)
        #   result["step3_no_emoticons"]에는 “이모티콘 제거된 텍스트”,
        #   result["emotion_chunks"], result["extracted_emoticons"]가 그대로 담겨있음

        # 3) msg에 필드 세팅
        msg["cleaned_text"]       = result["step3_no_emoticons"]
        msg["emotion_chunks"]     = result["emotion_chunks"]
        # msg["extracted_emoticons"] = result["extracted_emoticons"]
    return msgs

if __name__ == "__main__":
    import json
    
    # 📂 split_0.json 불러오기 (pipeline.py를 위해 주석 처리)
    with open("datasets/split_1.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🧼 전처리 수행 및 결과 추가
    for item in data:
        if "utterance" in item:
            result = full_preprocess(item["utterance"])
            item["cleaned"] = result["step3_no_emoticons"]
            item["emotion_chunks"] = result["emotion_chunks"]
            item["extracted_emoticons"] = result["extracted_emoticons"]
            
    # 💾 결과 저장 (overwrite or new file)
    with open("datasets/split_0_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ 전처리 완료: datasets/split_0_cleaned.json 저장됨")