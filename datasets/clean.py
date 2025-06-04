import sys
import os
import json

# 🔧 전처리 폴더 경로 추가
sys.path.append(os.path.join(os.getcwd(), "전처리"))

from chat import full_preprocess

# 📂 split_1.json 불러오기
with open("datasets/sample_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🧼 전처리 수행 및 결과 필드 추가
for item in data:
    if "utterance" in item:
        result = full_preprocess(item["utterance"])
        item["cleaned"] = result.get("step3_no_emoticons", "")
        item["emotion_chunks"] = result.get("emotion_chunks", {})
        item["extracted_emoticons"] = result.get("extracted_emoticons", [])

# 💾 저장: cleaned 버전 파일로
output_file = "datasets/sample_cleaned.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 전처리 완료: {output_file} 저장됨")