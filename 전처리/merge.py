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
def run_merge(filepath):
    msgs = parse_kakao_txt_with_ordered_merge(filepath)
    return msgs

# ✅ 테스트용
if __name__ == "__main__":
    clean_sentences = run_merge("datasets/KakaoTalk_20250515_0053_22_930_유정유정.txt")
    for s in clean_sentences[:5]:
        print(s)
        
# ✅ 실행 (직접 해당 파일을 실행할 때만 print 되도록 처리)
if __name__ == "__main__":
    msgs = parse_kakao_txt_with_ordered_merge("datasets/KakaoTalk_20250515_0053_22_930_유정유정.txt")
    for msg in msgs:
        print(f"{msg['timestamp']} {msg['speaker']} → {msg['text']}")
