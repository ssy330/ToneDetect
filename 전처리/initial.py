import re

# 축약어 사전
initial_abbr_dict = {
    'ㅇㅇ': '응',
    'ㅇㅋ': '알겠어',
    'ㄱㅊ': '괜찮아',
    'ㄴㄴ': '아니',
    'ㄱㄱ': '가자',
    'ㅂㅂ': '잘가',
    'ㅅㄱ': '수고해',
    'ㅁㄹ': '몰라',
    'ㅇㅈ': '인정',
    'ㅎㅇ': '안녕',
    'ㅁㅊ': '미친',
    'ㅈㅅ': '미안',
    'ㅊㅋ': '축하',
    'ㄷㄷ': '덜덜'
}

# 축약어 치환 함수
def replace_initial_abbr(text):
    def replace_match(match):
        word = match.group()
        for abbr, full in initial_abbr_dict.items():
            # 정확히 축약어가 단독으로 있거나, 단어 중간에 정확히 포함될 때만 치환
            if word == abbr:
                return full
            elif abbr in word and re.fullmatch(rf'[^ㄱ-ㅎ]*{abbr}[^ㄱ-ㅎ]*', word):
                return word.replace(abbr, full)
        return word  # 바꿀 게 없으면 그대로 반환

    # 한글 단어 단위로만 나누기 (기호나 영어, 숫자는 유지)
    return re.sub(r'[ㄱ-ㅎㅏ-ㅣ가-힣]+', replace_match, text)

# ✅ 카카오톡 내보내기 형식용 run_initial
def run_initial(raw_filepath):
    msgs = []
    date_pattern = re.compile(r"-{7,} (\d{4}년 \d{1,2}월 \d{1,2}일) .+ -{7,}")
    msg_pattern = re.compile(r"\[(.+?)\] \[(오전|오후) (\d+):(\d+)\] (.+)")

    current_date = None

    with open(raw_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 날짜 줄 처리
            date_match = date_pattern.match(line)
            if date_match:
                current_date = date_match.group(1)
                continue

            # 메시지 줄 처리
            msg_match = msg_pattern.match(line)
            if msg_match and current_date:
                speaker, period, hour, minute, text = msg_match.groups()

                # timestamp는 문자열로 우선 저장 (원하시면 datetime 변환 가능)
                timestamp_str = f"{current_date} {period} {hour}:{minute}"

                msg = {
                    "timestamp": timestamp_str,
                    "speaker": speaker,
                    "text": text
                }
                msgs.append(msg)

    return msgs

# 예시 실행
if __name__ == "__main__":
    samples = [
        "ㅇㅇ ㄱㄱ ㅊㅊ",
        "내일ㄱㄱ",
        "ㅇㅇㅎㅇ",
        "ㄱㄱㄱ ㄴㄴㅇㄴ",
        "오늘도ㅅㄱ!"
    ]

    for s in samples:
        print(f"입력: {s}")
        print(f"출력: {replace_initial_abbr(s)}\n")
