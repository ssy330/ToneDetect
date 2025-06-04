import re

# ✅ 1. 자모형 이모티콘 (ㅇㅅㅇ, ㅇ_ㅇ, ㅇ.ㅇ)
RE_HANGUL_FACE = re.compile(r'^[ㅇㅁㄴㄱㅎ]{1}[ㅅㅈ_\.]{1}[ㅇㅁㄴㄱㅎ]{1}$')

# ✅ 2. 기호형 이모티콘 (:), :-), :3 등)
RE_SYMBOL_FACE = re.compile(r'^[:;=~^][\-\'oO]?[)\(\|DdpP/<3\\]$')

# ✅ 3. 반복 눈모양 (ㅡㅡ, ㅇㅇ, ㅍㅍ 등)
RE_REPEAT_EYES = re.compile(r'^[ㅇㅡㅍ]{2,3}$')

# ✅ 4. 대칭 복합형 이모티콘 (>_<, ^_^, -_-, T_T, O_O 등)
RE_COMPLEX_FACE = re.compile(r'^[><\^@TtOo0\-]{1,2}[_\.\-]?[><\^@TtOo0\-]{1,2}$')

# ✅ 5. 유니코드 이모지 (그림 이모티콘 😊🔥💥 등)
RE_UNICODE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # 심볼, 날씨, 음식 등
    "\U0001F600-\U0001F64F"  # 얼굴, 감정 표현
    "\U0001F680-\U0001F6FF"  # 교통, 물건 등
    "\U0001F700-\U0001F77F"
    "\U0001F900-\U0001F9FF"
    "\u2600-\u26FF"          # 기호 ☀, ☔ 등
    "\u2700-\u27BF"          # 체크, 펜 등
    "]"
)

# ✅ 이모티콘 여부 판단 함수 (토큰 단위)
def is_emoticon(token: str) -> bool:
    return (
        RE_HANGUL_FACE.fullmatch(token) or
        RE_SYMBOL_FACE.fullmatch(token) or
        RE_REPEAT_EYES.fullmatch(token) or
        RE_COMPLEX_FACE.fullmatch(token) or
        bool(RE_UNICODE_EMOJI.search(token))  # “토큰 안에 유니코드 이모지가 하나라도 있으면 True”
    )

# ✅ 이모티콘만 추출 (토큰 단위)
def extract_emoticons(text: str) -> list:
    # 공백으로 토큰화 후, is_emoticon(token)이 True인 토큰만 반환
    tokens = re.findall(r'[^\s]+', text)
    return [tok for tok in tokens if is_emoticon(tok)]

# ✅ 이모티콘 제거한 텍스트 반환 (토큰 단위)
def remove_emoticons(text: str) -> str:
    tokens = re.findall(r'[^\s]+', text)
    return ' '.join([tok for tok in tokens if not is_emoticon(tok)])

# ───────────────────────────────────────────────────────────────────────────
# “토큰 단위로 뽑은 문자열” 안에서 **순수 이모티콘 문자만** 골라내기 위한 헬퍼
# ───────────────────────────────────────────────────────────────────────────
def extract_only_emoji(raw_str: str) -> list:
    """
    raw_str 안에 “여러 글자(예: 고마어🥰🥰)”가 붙어 있을 때,
    유니코드 이모지(😊🔥 등)만 골라서 리스트로 돌려줍니다.
    예: "고마어🥰🥰" → ["🥰", "🥰"]
    """
    # “단일 이모지” 유니코드 범위를 정리한 정규식
    emoji_pattern = re.compile(
        r'['
        r'\U0001F300-\U0001F5FF'
        r'\U0001F600-\U0001F64F'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F'
        r'\U0001F780-\U0001F7FF'
        r'\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF'
        r'\u2600-\u26FF'
        r'\u2700-\u27BF'
        r']'
    )
    # findall 하면 "고마어🥰🥰" 속의 “🥰”만 ["🥰", "🥰"]로 반환됩니다.
    return emoji_pattern.findall(raw_str)

# ───────────────────────────────────────────────────────────────────────────
# pipeline 에서 호출할 run_emoticon 정의 (업데이트된 버전)
# ───────────────────────────────────────────────────────────────────────────
def run_emoticon(msgs):
    for msg in msgs:
        # ⚡ “원문(이모티콘 포함된 상태)”을 보존하기 위해 복사본 생성
        original = msg["text"]
        msg["original_text"] = original

        # 1) 토큰 단위로 뽑아서, 이모티콘 토큰 리스트 얻기
        raw_emoticons = extract_emoticons(original)
        #    예: raw_emoticons == ["고마어🥰🥰", "알써😆😆", "보자😂😂❤", …]

        # 2) 토큰별로 “순수 이모티콘 문자”만 분리하여 하나의 리스트에 모으기
        emoticons = []
        for tok in raw_emoticons:
            emoticons.extend(extract_only_emoji(tok))
        #    결과 예: emoticons == ["🥰","🥰","😆","😆","😂","😂","❤"]

        # 3) 텍스트에서 토큰 단위로 이모티콘 포함된 단어를 통째로 제거
        cleaned = remove_emoticons(original)

        # 4) msg에 두 가지 정보를 모두 저장
        msg["text"] = cleaned
        msg["extracted_emoticons"] = emoticons

        # ✅ 디버그 출력
        print(f"[run_emoticon] speaker={msg['speaker']}, extracted_emoticons={emoticons}")

    return msgs


if __name__ == "__main__":
    sample = "진짜 너무하네 ㅇㅅㅇ >_< 헐 :) ㅡㅡ ^_^ ㅍㅍ O_O :'(" \
             " 그리고 이모지도 있어요 ❤️🔥🙂ㅋㅋㅋㅠㅠ"

    print("🟣 원본 텍스트:", sample)
    print("🟣 토큰 단위 이모티콘 추출:", extract_emoticons(sample))
    #   → 예: ['ㅇㅅㅇ', '>_<', ':)', '^^', 'ㅍㅍ', 'O_O', '❤️', '🔥', '🙂', 'ㅋㅋㅋ', 'ㅠㅠ']
    print("🟢 순수 이모티콘만 분리:", 
          [extract_only_emoji(tok) for tok in extract_emoticons(sample)])
    #   → 예: [[], [], [], ..., ['❤️'], ['🔥'], ['🙂'], [], ['ㅠ','ㅠ']]
    print("🟢 최종 이모티콘 리스트:", 
          [e for tok in extract_emoticons(sample) for e in extract_only_emoji(tok)])
    #   → 예: ['❤️', '🔥', '🙂', 'ㅠ', 'ㅠ']
    print("🟢 정제된 문장:", remove_emoticons(sample))