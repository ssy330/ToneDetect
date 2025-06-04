# app.py (보완본)

import os, sys, uuid, glob, re
import torch, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.append(os.path.join(os.getcwd(), "전처리"))

# 전처리 모듈 import
from initial    import run_initial      # raw 파일 읽기 + 기본 전처리
from emoticon   import run_emoticon
from clean      import run_clean        # 일반 텍스트 클린징
from chat       import run_chat, full_preprocess # 카톡 말투·축약어 처리
from merge      import run_merge        # 화자별로 flatten + 최종 합치기

# ✅ 진행률 저장용 (전역)
progress_data = {
    "progress": 0
}

plt.rcParams['font.family'] = 'AppleGothic'

# Flask 앱 생성
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GRAPH_FOLDER'] = 'static'

# 모델 로드
ADAPTER_PATH = "Models/ToneDetect_adapter"
BASE_MODEL_NAME = "beomi/kcbert-base"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=5)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# 예측 함수
def predict_style(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = torch.argmax(logits, dim=1).item()
    return pred_class

# 라벨 매핑
label_map = {
    0: "chat_emoticon(이모티콘 자주 쓰는 말투)",
    1: "elder_speech(어르신 말투)",
    2: "formal(격식있는 말투)",
    3: "informal(친근한 말투)",
    4: "soft_polite(부드럽고 상냥한 말투)"
}

# 루트 페이지
@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    top_styles = None
    graph_filename = None
    speaker_stats = None

    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="파일을 선택해주세요!")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # 전처리 파이프라인 실행 전에 txt 파일인지 체크
        if not filepath.endswith('.txt'):
            return render_template('index.html', error="KakaoTalk 텍스트(.txt) 파일만 지원합니다!")
        
        ### ✅ ✅ 기존 그래프 삭제 (매번 분석 시작 시!)
        old_graphs = glob.glob(os.path.join(app.config['GRAPH_FOLDER'], "*.png"))
        for old_file in old_graphs:
            os.remove(old_file)
            
        # ──────────────── 전처리 파이프라인 실행(단계별로) ────────────────
        # 🔸🔸 단계별 호출 시작 🔸🔸

        msgs = run_initial(filepath)
        msgs = run_emoticon(msgs) 
        # msgs = run_emotion(msgs)
        msgs = run_clean(msgs)
        print("▶ run_clean 후 sample msg:", msgs[0])
        msgs = run_chat(msgs)
            
        # ✅ 분석 시작 시 진행률 초기화
        progress_data["progress"] = 0

        # 분석
        speaker_style_counts = defaultdict(lambda: defaultdict(int))
        total_msgs = len(msgs)

        for idx, msg in enumerate(msgs):
            speaker = msg['speaker']
            text = msg['cleaned_text']
            pred = predict_style(text)
            style_name = label_map[pred]
            speaker_style_counts[speaker][style_name] += 1

            # ✅ 진행률 업데이트
            progress_data["progress"] = int((idx + 1) / total_msgs * 100)

        # DataFrame 변환
        data = []
        top_styles = {}
        for speaker, style_counts in speaker_style_counts.items():
            row = {'speaker': speaker}
            row.update(style_counts)
            data.append(row)

            # 🟢 가장 많이 사용한 말투
            if style_counts:
                top_style = max(style_counts.items(), key=lambda x: x[1])[0]
                top_styles[speaker] = top_style
            else:
                top_styles[speaker] = "데이터 없음"

        # 1️⃣ 원본 DataFrame 유지
        df = pd.DataFrame(data).fillna(0)

        # 2️⃣ 표용 복사본 따로 만들기
        df_html = df.copy()

        html_columns = []
        for col in df_html.columns:
            if col == 'speaker':
                html_columns.append('speaker')
            else:
                eng, kor = col.split('(')
                kor = kor.rstrip(')')
                html_columns.append(f"{eng}<br>({kor})")

        df_html.columns = html_columns
        table_html = df_html.to_html(classes='data', index=False, escape=False)
        
        # 3️⃣ 그래프용 → 원본 df 사용
        df.set_index('speaker', inplace=True)

        # 🟢 그래프 저장 준비
        os.makedirs(app.config['GRAPH_FOLDER'], exist_ok=True)
        graph_filename = f"{uuid.uuid4().hex}.png"
        graph_path = os.path.join(app.config['GRAPH_FOLDER'], graph_filename)

        # 3️⃣ 그래프용 → 원본 df 사용
        df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
        plt.title("📊 화자별 말투 스타일 분포 (Stacked Bar)", fontsize=14)
        plt.xlabel("화자", fontsize=12)
        plt.ylabel("문장 수", fontsize=12)
        plt.xticks(rotation=30)
        plt.legend(title='말투 스타일', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()
        
        
        # ✅ 🔥 감정 문자열 / 이모티콘 분석용 speaker_stats 생성
        speaker_emotion_counter = defaultdict(Counter)
        speaker_emotion_examples = defaultdict(lambda: defaultdict(list))

        speaker_emo_counter = defaultdict(Counter)
        speaker_emo_examples = defaultdict(lambda: defaultdict(list))

        for item in msgs:
            speaker = item["speaker"]
            
            # ★ 예시 문장은 반드시 “원문(original_text)”을 사용해야 이모티콘/이모지가 보입니다.
            example_text = item.get("original_text", "")

            # 감정 문자열
            already_added_chunks = set()
            for chunk_list in item.get("emotion_chunks", {}).values():
                for chunk in chunk_list:
                    speaker_emotion_counter[speaker][chunk] += 1
                    if chunk not in already_added_chunks:
                        if len(speaker_emotion_examples[speaker][chunk]) < 3:
                            speaker_emotion_examples[speaker][chunk].append(example_text)
                        already_added_chunks.add(chunk)

            # 이모티콘
            already_added_emoticons = set()
            # run_emoticon에서 뽑아 둔 “순수 토큰”을 그대로 사용
            for emo in item.get("extracted_emoticons", []):
                speaker_emo_counter[speaker][emo] += 1
                if emo not in already_added_emoticons:
                    if len(speaker_emo_examples[speaker][emo]) < 3:
                        speaker_emo_examples[speaker][emo].append(example_text)
                    already_added_emoticons.add(emo)

        # ✅ speaker_stats 딕셔너리 구성
        speaker_stats = {}

        for speaker in sorted(set(speaker_emotion_counter) | set(speaker_emo_counter)):
            speaker_stats[speaker] = {
                "emotion_chunks": [],
                "extracted_emoticons": []
            }

            # 감정 문자열
            top_chunks = speaker_emotion_counter[speaker].most_common(3)
            for chunk, count in top_chunks:
                speaker_stats[speaker]["emotion_chunks"].append({
                    "chunk": chunk,
                    "count": count,
                    "examples": speaker_emotion_examples[speaker][chunk]
                })

            # 이모티콘
            top_emoticons = speaker_emo_counter[speaker].most_common(3)
            for emo, count in top_emoticons:
                speaker_stats[speaker]["extracted_emoticons"].append({
                    "emoji": emo,
                    "count": count,
                    "examples": speaker_emo_examples[speaker][emo]
                })
            
        # ✅ 디버그 출력
        print("=== speaker_stats 최종 결과 ===")
        import pprint
        pprint.pprint(speaker_stats)
        
        msgs = run_merge(msgs)
                
        # ✅ 최종 진행률 100%로 설정 (완료 표시)
        progress_data["progress"] = 100

    return render_template('index.html',
                           table_html=table_html,
                           top_styles=top_styles,
                           graph_filename=graph_filename,
                           speaker_stats=speaker_stats)
    
# ✅ 진행률 조회용 route 추가
@app.route('/progress', methods=['GET'])
def progress():
    return {"progress": progress_data["progress"]}

# 실행
if __name__ == '__main__':
    app.run(debug=True)