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

# ─────────── 여기에 LDA용 라이브러리 추가 ───────────
from konlpy.tag import Okt
from gensim import corpora
from gensim.models import LdaModel


# ───────── 워드클라우드 관련 라이브러리 추가 ─────────
from wordcloud import WordCloud

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

# ───────── 워드클라우드용 기본 설정 ─────────
# Mac: 기본 AppleGothic, Windows: 'C:/Windows/Fonts/malgun.ttf' 등 환경에 맞게 수정하세요.
FONT_PATH = '/Library/Fonts/AppleGothic.ttf'

def generate_topic_wordclouds(lda_model, dictionary, num_topics=5, topn=30):
    """
    lda_model: gensim.models.LdaModel 객체
    dictionary: LDA 학습에 사용된 Gensim Dictionary
    num_topics: 워드클라우드로 만들 토픽 개수
    topn: 토픽별 상위 몇 개 단어를 워드클라우드에 반영할지
    반환값: 만든 이미지 파일명 리스트 (예: ['topic_1.png', 'topic_2.png', ...])
    """
    # 'static/wordclouds' 폴더를 만들어두고, 그곳에 이미지를 저장하겠습니다.
    wc_folder = os.path.join(app.config['GRAPH_FOLDER'], 'wordclouds')
    os.makedirs(wc_folder, exist_ok=True)

    filenames = []
    for topic_id in range(num_topics):
        # 각 토픽에서 topn 단어+가중치 추출
        topic_terms = lda_model.show_topic(topic_id, topn=topn)
        # 예: [("단어1", 0.05), ("단어2", 0.03), ...]
        freq_dict = {word: float(weight) for word, weight in topic_terms}

        # 워드클라우드 생성
        wc = WordCloud(
            background_color='white',
            font_path=FONT_PATH,
            width=800,
            height=400
        ).generate_from_frequencies(freq_dict)

        # 파일명 예시: 'topic_1.png', 'topic_2.png', ...
        file_name = f"topic_{topic_id+1}.png"
        save_path = os.path.join(wc_folder, file_name)

        # pyplot 없이 직접 저장
        wc.to_file(save_path)
        filenames.append(os.path.join('wordclouds', file_name))  # 템플릿에서 사용할 경로 (static/wordclouds/...)
    return filenames

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
    lda_topics    = None  # LDA 토픽 결과를 담을 변수
    wc_filenames   = None  # 워드클라우드 이미지 파일명을 담을 리스트


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
            
        # (2) static/wordclouds 폴더가 있다면, 그 안의 PNG도 전부 삭제
        wc_folder = os.path.join(app.config['GRAPH_FOLDER'], 'wordclouds')
        if os.path.isdir(wc_folder):
            old_wcs = glob.glob(os.path.join(wc_folder, "*.png"))
            for old_wc in old_wcs:
                os.remove(old_wc)
            
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
        
        merged_msgs = run_merge(msgs)

        # ───────────── run_merge 직후 디버깅 코드 시작 ─────────────
        print("▶ run_merge() 후 merged_msgs 샘플 (총 개수:", len(merged_msgs), "개)")

        import pprint
        pprint.pprint(merged_msgs[:5])

        print("\n▶ run_merge() 후 merged_msgs 각 항목 상세보기 (최초 5개)")
        for i, m in enumerate(merged_msgs[:5], start=1):
            print(f"--- 메시지 #{i} ---")
            print(f"timestamp : {m.get('timestamp')}")
            print(f"speaker   : {m.get('speaker')}")
            print(f"text      : {m.get('text')}")
            remaining = {k: v for k, v in m.items() if k not in ['timestamp','speaker','text']}
            print("그 외 필드:", remaining)
            print()
        # ───────────── run_merge 직후 디버깅 코드 끝 ─────────────
        
        # ─────────────────────────────────────────────────────────────
        # (1) “최종 정제된 문장” 리스트 생성 (run_merge 이후)
        texts = [
            item["text"]
            for item in merged_msgs
            if item.get("text") and item["text"].strip()
        ]

        # (2) 형태소 분석기로 각 문장에서 명사만 추출 → 토큰화된 문장 리스트
        okt = Okt()
        tokenized_texts = [okt.nouns(txt) for txt in texts]

        # ─────────────── LDA 실행 전 예외 처리 ───────────────
        # tokenized_texts 자체가 비어 있거나,
        # tokenized_texts 내의 모든 요소가 빈 리스트일 때 LDA를 실행하면 오류 발생하므로
        # 이 경우 lda_topics를 빈 리스트로 설정하고 건너뜁니다.
        if not tokenized_texts or all(len(tokens) == 0 for tokens in tokenized_texts):
            # LDA를 돌릴 문장이 없으므로, 빈 결과를 할당
            lda_topics = []
            wc_filenames = []
        else:
            # (3) Gensim Dictionary + Corpus(BOW) 생성
            dictionary = corpora.Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

            # (4) LDA 모델 학습 (토픽 수·패스 수는 필요에 맞게 조절 가능)
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=5,
                passes=10,
                random_state=42
            )

            # (5) 토픽 결과 추출
            topics = lda_model.print_topics(num_words=5)
            lda_topics = []
            for idx, topic_string in topics:
                lda_topics.append({
                    "topic_id": idx + 1,    # 화면에 보여줄 때는 1부터 시작
                    "keywords": topic_string
                })
        # ─────────────────────────────────────────────────────────────
                
        # ───────────────── 워드클라우드 생성 코드 시작 ─────────────────

        # 워드클라우드를 저장할 디렉토리 (static/wordclouds)
        wc_folder = os.path.join(app.config['GRAPH_FOLDER'], 'wordclouds')
        os.makedirs(wc_folder, exist_ok=True)

        # 한글 폰트 경로: 환경에 맞게 수정해주세요.
        # macOS 예시: '/Library/Fonts/AppleGothic.ttf'
        # Windows 예시: 'C:/Windows/Fonts/malgun.ttf'
        FONT_PATH = '/Library/Fonts/AppleGothic.ttf'

        # 토픽 개수 (lda_model.num_topics) 만큼 워드클라우드 생성
        num_topics = lda_model.num_topics
        wc_filenames = []  # 생성된 이미지 파일명을 차례대로 저장할 리스트

        for topic_id in range(min(3, num_topics)):
            # 각 토픽에서 상위 30개 단어+가중치 추출
            topic_terms = lda_model.show_topic(topic_id, topn=30)
            # topic_terms 예: [('정산', 0.05), ('요청', 0.03), …]

            # 워드클라우드에 넘길 빈도 사전 생성
            freq_dict = { word: float(weight) for word, weight in topic_terms }

            # WordCloud 객체 생성 및 빈도 사전 반영
            wc = WordCloud(
                background_color='white',
                font_path=FONT_PATH,
                width=800,
                height=400
            ).generate_from_frequencies(freq_dict)

            # 파일명: topic_1.png, topic_2.png, … 형태
            file_name = f"topic_{topic_id+1}.png"
            save_path = os.path.join(wc_folder, file_name)

            # 이미지 파일로 저장
            wc.to_file(save_path)

            # 템플릿에 넘겨줄 때는 'wordclouds/topic_1.png' 경로로 사용
            wc_filenames.append(os.path.join('wordclouds', file_name))
        # ───────────────── 워드클라우드 생성 코드 끝 ─────────────────
        
        # ✅ 최종 진행률 100%로 설정 (완료 표시)
        progress_data["progress"] = 100

    return render_template('index.html',
                           table_html=table_html,
                           top_styles=top_styles,
                           graph_filename=graph_filename,
                           speaker_stats=speaker_stats,
                           lda_topics = lda_topics,
                           wc_filenames = wc_filenames)
    
# ✅ 진행률 조회용 route 추가
@app.route('/progress', methods=['GET'])
def progress():
    return {"progress": progress_data["progress"]}

# 실행
if __name__ == '__main__':
    app.run(debug=True)