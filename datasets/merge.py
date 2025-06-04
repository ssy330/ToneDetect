import json
import re
from datetime import datetime

# 문장 종결 판단
def is_sentence_end(text):
    return bool(re.search(r"(?:[.!?ㅋㅎㅠㅜ~;]){1,}$", text.strip())) or text.endswith(("요", "임", "함", "다", "네"))

# 병합 로직 (정렬 없이 순서 유지)
def merge_utterances_by_minute(data):
    result = []
    prev = None  # 직전 항목 저장

    for item in data:
        time_str = item["time"]
        participant = item["participantID"]
        utterance = item["utterance"]
        time_key = datetime.strptime(time_str, "%H:%M:%S").strftime("%H:%M")

        if prev:
            prev_time_key = datetime.strptime(prev["time"], "%H:%M:%S").strftime("%H:%M")
            same_person = prev["participantID"] == participant
            same_minute = prev_time_key == time_key
            prev_text = prev["utterance"]

            if same_person and same_minute and not is_sentence_end(prev_text):
                # 병합
                prev["utterance"] += " " + utterance
                continue
            else:
                result.append(prev)

        # prev 초기화
        prev = {
            "time": time_str,
            "participantID": participant,
            "utterance": utterance
        }

    # 마지막 항목 추가
    if prev:
        result.append(prev)

    return result

# 실행 함수
def run_merge_json(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    merged = merge_utterances_by_minute(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ 병합 완료: {output_path}")

# 실행 예시
if __name__ == "__main__":
    run_merge_json("datasets/sample_preprocessing.json", "datasets/sample_merged.json")
