import json
import re
import random

# 파일 경로
input_path = "datasets/sample.json"
output_path = "datasets/sample_preprocessing.json"

# 이름 리스트
name_list = ["소연", "상권", "유정", "서연", "혜성", "선혁", "가영", "요한", "혜명", "영학", "주찬", "주은"]

# 이름 치환 함수
def replace_name(text):
    return text.replace("#@이름#", random.choice(name_list))

# 이름 외의 #@...# 패턴
pattern_other_tags = re.compile(r"#@(?!(이름)#)[^#]+#")

# JSON 불러오기
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

utterances_list = []

for item in raw_data.get("data", []):
    for utter in item.get("body", []):
        text = utter.get("utterance", "")
        time = utter.get("time", "")
        participant = utter.get("participantID", "")

        # 이름 이외의 특수 태그가 있으면 건너뛰기
        if pattern_other_tags.search(text):
            continue

        # 이름 치환
        cleaned_text = replace_name(text)

        # 결과 저장
        utterances_list.append({
            "time": time,
            "participantID": participant,
            "utterance": cleaned_text.strip()
        })

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(utterances_list, f, ensure_ascii=False, indent=2)

print(f"{len(utterances_list)}개 문장이 정제되어 저장되었습니다.")
