# pipeline.py

import os
import sys

# 1) 전처리 모듈 경로를 시스템 패스에 추가
#    (이렇게 하면 아래에서 import 전처리 모듈들이 정상적으로 로드됩니다)
sys.path.append(os.path.join(os.getcwd(), "전처리"))

# 2) 각 단계별 전처리 함수를 import
from initial    import run_initial      # raw 파일 읽기 + 기본 전처리
from clean      import run_clean        # 일반 텍스트 클린징
from emoticon   import run_emoticon     # 이모티콘/이모지 처리
from emotion    import run_emotion      # 감정 표현 처리
from chat       import run_chat         # 카톡 말투·축약어 처리
from merge      import run_merge        # 화자별로 flatten + 최종 합치기

def run_pipeline(raw_filepath: str):
    """
    전체 전처리 파이프라인을 한 번에 실행하는 함수.

    1) initial 단계: raw .txt 파일을 읽어서 원본 메시지 리스트 생성
    2) clean 단계: 일반 텍스트 클린징
    3) emoticon 단계: 이모티콘/이모지 처리
    4) emotion 단계: 감정 표현 처리
    5) chat 단계: 카톡 고유 말투 처리
    6) merge 단계: 화자별로 flatten 해서 최종 { speaker, text } 리스트 반환

    리턴: List[Dict], 각 항목에는 최소한 { "speaker": str, "text": str }가 담겨있어야 함.
    """

    # ─────────────────────────────────────────────────────────────
    # 1) initial 단계: 원본 .txt 파일에서 메시지 객체 리스트를 생성
    #    (예: [{ "timestamp": "...", "speaker": "...", "text": "..." }, ...])
    msgs = run_initial(raw_filepath)

    # ─────────────────────────────────────────────────────────────
    # 2) clean 단계: 일반 클린징 (불필요한 특수문자 제거, 공백 정리 등)
    msgs = run_clean(msgs)

    # ─────────────────────────────────────────────────────────────
    # 3) emoticon 단계: 이모티콘/이모지 처리 (필요하다면):
    msgs = run_emoticon(msgs)

    # ─────────────────────────────────────────────────────────────
    # 4) emotion 단계: 감정 표현 추출/정제 (필요하다면):
    msgs = run_emotion(msgs)

    # ─────────────────────────────────────────────────────────────
    # 5) chat 단계: 카톡 말투(줄임말, 특이 맞춤법 등) 처리 (필요하다면):
    msgs = run_chat(msgs)

    # ─────────────────────────────────────────────────────────────
    # 6) merge 단계: 화자별로 flatten + 최종 합치기
    #    (예: [{ "speaker": "유정유정", "text": "안녕!" }, ... ] 형태로 반환)
    msgs = run_merge(msgs)

    return msgs