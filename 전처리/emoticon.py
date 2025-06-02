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

# ✅ 이모티콘 여부 판단 함수
def is_emoticon(token: str) -> bool:
    return (
        RE_HANGUL_FACE.fullmatch(token) or
        RE_SYMBOL_FACE.fullmatch(token) or
        RE_REPEAT_EYES.fullmatch(token) or
        RE_COMPLEX_FACE.fullmatch(token) or
        bool(RE_UNICODE_EMOJI.search(token))  # ✅ 그림 이모지 포함
    )

# ✅ 이모티콘만 추출
def extract_emoticons(text: str) -> list:
    tokens = re.findall(r'[^\s]+', text)
    return [tok for tok in tokens if is_emoticon(tok)]

# ✅ 이모티콘 제거한 텍스트 반환
def remove_emoticons(text: str) -> str:
    tokens = re.findall(r'[^\s]+', text)
    return ' '.join([tok for tok in tokens if not is_emoticon(tok)])

if __name__ == "__main__":
    sample = "진짜 너무하네 ㅇㅅㅇ >_< 헐 :) ㅡㅡ ^_^ ㅍㅍ O_O :'(" \
             " 그리고 이모지도 있어요 ❤️🔥🙂ㅋㅋㅋㅠㅠ"

    print("🟣 이모티콘 추출:", extract_emoticons(sample))
    print("🟢 정제된 문장:", remove_emoticons(sample))