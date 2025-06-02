import json
from chat import full_preprocess

# 📂 split_0.json 불러오기
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