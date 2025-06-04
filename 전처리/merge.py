import re
from datetime import datetime
from collections import deque

def parse_time(date_str, period, hour, minute):
    dt = datetime.strptime(date_str, "%Y년 %m월 %d일")
    hour = int(hour)
    if period == "오후" and hour != 12:
        hour += 12
    elif period == "오전" and hour == 12:
        hour = 0
    return dt.replace(hour=hour, minute=int(minute))

def is_sentence_end(text):
    return bool(re.search(r"(?:[.!?ㅋㅎㅠㅜ~]){1,}$", text.strip()))

def parse_kakao_txt_with_ordered_merge(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()[2:]

    date_pattern = re.compile(r"-{7,} (\d{4}년 \d{1,2}월 \d{1,2}일) .+ -{7,}")
    msg_pattern = re.compile(r"\[(.+?)\] \[(오전|오후) (\d+):(\d+)\] (.+)")

    current_date = None
    result = []
    buffer = {}
    order = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        date_match = date_pattern.match(line)
        if date_match:
            current_date = date_match.group(1)
            continue

        msg_match = msg_pattern.match(line)
        if not msg_match or not current_date:
            continue

        speaker, period, hour, minute, utterance = msg_match.groups()
        timestamp = parse_time(current_date, period, hour, minute)

        prev = buffer.get(speaker)
        if prev:
            prev_time, prev_text, prev_order = prev
            gap = (timestamp - prev_time).total_seconds()
            if gap <= 30 and not is_sentence_end(prev_text):
                # 병합
                merged = prev_text + " " + utterance
                buffer[speaker] = (timestamp, merged, prev_order)
                continue
            else:
                result.append({
                    "speaker": speaker,
                    "text": prev_text,
                    "timestamp": prev_time,
                    "order": prev_order
                })

        # 새 발화 시작
        buffer[speaker] = (timestamp, utterance, order)
        order += 1

    # 남은 메시지 처리
    for speaker, (timestamp, text, order_index) in buffer.items():
        result.append({
            "speaker": speaker,
            "text": text,
            "timestamp": timestamp,
            "order": order_index
        })

    # 정렬: 시간 → 등장순
    result.sort(key=lambda x: (x["timestamp"], x["order"]))
    return result

# ✅ 새로 추가: pipeline용 run_merge()
def run_merge(msgs):
    # 예시: 화자별로 flatten 처리
    merged_msgs = []
    for msg in msgs:
        merged_msg = {
            "speaker": msg["speaker"],
            "text": msg["text"]
        }
        merged_msgs.append(merged_msg)
    return merged_msgs

# ✅ 테스트용
if __name__ == "__main__":
    filepath = "datasets/KakaoTalk_20250515_0053_22_930_유정유정.txt"

    # 1️⃣ 파싱 단계
    msgs = parse_kakao_txt_with_ordered_merge(filepath)

    # 2️⃣ run_merge 실행
    clean_sentences = run_merge(msgs)

    # 3️⃣ 출력
    for s in clean_sentences[:5]:
        print(s)

    # 추가 테스트: 원본 메시지 출력
    print("\n--- 원본 parse 결과 ---")
    for msg in msgs[:5]:
        print(f"{msg['timestamp']} {msg['speaker']} → {msg['text']}")
        print(f"{msg['timestamp']} {msg['speaker']} → {msg['text']}")
