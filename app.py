import os
import tempfile
import gc
import re
import io
import streamlit as st
import pandas as pd
from faster_whisper import WhisperModel

# ==========================================
# 1. 頁面與 UI 基本配置
# ==========================================
st.set_page_config(
    page_title="台灣政治言談語音轉文字系統 (faster-whisper)",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 台灣政治言談語音轉文字與文本分析工具")
st.markdown("本系統採用 **faster-whisper (int8 CPU 量化)**，支援台灣議會轉檔、關鍵詞敘述統計、百萬詞頻 (PMM)、共現分析與 TF-IDF 關鍵字提取。")

# 嘗試載入分析套件
try:
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    st.error("未安裝必要套件，請於 requirements.txt 加入 jieba 與 scikit-learn。")

# ==========================================
# 2. 模型載入機制
# ==========================================
@st.cache_resource
def load_whisper_model():
    model_size = "medium"
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8",
        cpu_threads=4
    )
    return model

try:
    with st.spinner("正在載入語音辨識模型 (僅初次載入需數秒)..."):
        model = load_whisper_model()
    st.success("✅ 模型載入成功！")
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# ==========================================
# 3. 文本校正與分析核心函數
# ==========================================
def refine_taiwan_terms(text: str) -> str:
    if not text:
        return ""
    replacements = {
        r"陸委會": "陸委會",
        r"海基會": "海基會",
        r"中華民國": "中華民國",
        r"中國大陸": "中國大陸",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

def analyze_text(full_text: str, target_word: str, window_size: int = 5):
    """計算基礎統計、PMM、共現詞與 TF-IDF 前 10 大關鍵字"""
    clean_char_text = re.sub(r"[^\w]", "", full_text)
    total_chars = len(clean_char_text)
    
    # 斷詞處理
    words = [w for w in jieba.cut(full_text) if re.match(r"[\u4e00-\u9fa5a-zA-Z0-9]+", w) and len(w) > 1]
    total_words = len(words)
    
    if total_words == 0:
        return None

    # 1. 計算目標詞頻與百萬詞頻 (PMM)
    target_count = full_text.count(target_word) if target_word else 0
    pmm = (target_count / total_words) * 1_000_000 if total_words > 0 else 0.0

    # 2. 共現詞分析
    from collections import Counter
    co_occurrences = Counter()
    all_words = [w for w in jieba.cut(full_text) if re.match(r"[\u4e00-\u9fa5a-zA-Z0-9]+", w)]
    if target_word and target_count > 0:
        for i, word in enumerate(all_words):
            if word == target_word:
                start = max(0, i - window_size)
                end = min(len(all_words), i + window_size + 1)
                context = all_words[start:i] + all_words[i+1:end]
                for ctx_word in context:
                    if len(ctx_word) > 1 and ctx_word != target_word:
                        co_occurrences[ctx_word] += 1

    # 3. TF-IDF 關鍵字提取
    # 將全篇文本按句拆分作為文檔集以計算相對重要性
    sentences = [s.strip() for s in re.split(r"[。！？\n]", full_text) if len(s.strip()) > 5]
    top_tfidf_words = []
    if sentences:
        corpus = [" ".join(jieba.cut(s)) for s in sentences]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0)
        
        data = []
        for col, cost in enumerate(sums.A[0]):
            word = feature_names[col]
            if len(word) > 1 and not word.isdigit():
                data.append((word, cost))
        data.sort(key=lambda x: x[1], reverse=True)
        top_tfidf_words = data[:10]

    return {
        "total_chars": total_chars,
        "total_words": total_words,
        "target_count": target_count,
        "pmm": pmm,
        "co_occurrences": co_occurrences.most_common(10),
        "tfidf": top_tfidf_words
    }

# ==========================================
# 4. Prompt 設定與介面
# ==========================================
default_prompt = (
    "以下為中華民國台灣縣市議會與政壇討論質詢錄音。"
    "關鍵詞語：中華民國、台灣、臺灣、中國大陸、大陸、中國、兩岸關係、陸委會、中央政府、地方政府。"
    "請嚴格區分「台灣/臺灣」、「中華民國」、「中國大陸」、「中國」等不同政治實體稱謂。"
    "議事常見術語：總質詢、業務質詢、預算案、墊付案、三讀、附帶決議、請就座、時間暫停。"
    "常見單位與稱謂：議員、議長、縣長、市長、局長、處長、主計處、研考會、工務局、衛生局。"
)

with st.expander("⚙️ 高級設定：語意與政治名詞 Prompt 導引", expanded=False):
    custom_prompt = st.text_area("語意導引 Prompt：", value=default_prompt, height=120)

# ==========================================
# 5. 上傳提示卡片（長度與預估時間說明）
# ==========================================
st.info("""
⏱️ **音訊上傳限制與處理時間說明 (純 CPU 伺服器環境)：**
* **建議音訊長度：** 請控制在 **10 分鐘以內**（若檔期過長，建議分割後分段上傳）。
* **長度上限警示：** 超過 **20 分鐘** 的音訊有機率觸發伺服器 Timeout 逾時連線中斷。
* **預估處理時間：**
  * ⏱️ **1 ~ 3 分鐘音訊：** 約需 30 秒 ~ 1 分鐘
  * ⏱️ **5 ~ 10 分鐘音訊：** 約需 2 ~ 4 分鐘
  * ⏱️ **10 ~ 15 分鐘音訊：** 約需 5 ~ 8 分鐘
""")

uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

# ==========================================
# 6. 執行語音辨識與文本分析
# ==========================================
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 開始精準辨識與分析", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("語音辨識中，請稍候（依音訊長度需數十秒至數分鐘）..."):
            try:
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=7,
                    language="zh",
                    temperature=0.0,
                    condition_on_previous_text=True,
                    initial_prompt=custom_prompt
                )

                st.subheader("📝 辨識結果與逐字稿")
                full_text_list = []
                
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    for segment in segments:
                        start, end = segment.start, segment.end
                        clean_text = refine_taiwan_terms(segment.text)
                        full_text_list.append(clean_text)
                        st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {clean_text}")

                full_text = "\n".join(full_text_list)
                st.session_state["full_text"] = full_text
                st.session_state["file_name"] = uploaded_file.name

                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ 轉檔過程中發生錯誤：{e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                gc.collect()

# ==========================================
# 7. 文本分析與 CSV 導出模組
# ==========================================
if "full_text" in st.session_state and st.session_state["full_text"]:
    st.markdown("---")
    st.header("📊 逐字稿定量與關鍵詞文本分析")
    
    full_text = st.session_state["full_text"]

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        target_word = st.text_input("請輸入欲分析的政治關鍵詞：", value="台灣")
    with col_input2:
        window_size = st.slider("共現分析視窗範圍 (前後字數)：", min_value=2, max_value=10, value=5)

    if target_word:
        with st.spinner("正在計算詞頻、共現詞與 TF-IDF 關鍵字..."):
            jieba.add_word(target_word)
            jieba.add_word("中華民國")
            jieba.add_word("中國大陸")
            
            stats = analyze_text(full_text, target_word, window_size)

        if stats:
            # 1. 敘述統計指標看板
            st.subheader("1. 基礎敘述統計與百萬詞頻 (PMM)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總字數 (含標點)", f"{stats['total_chars']} 字")
            m2.metric("總詞數 (Segs)", f"{stats['total_words']} 詞")
            m3.metric(f"「{target_word}」出現次數", f"{stats['target_count']} 次")
            m4.metric(f"百萬詞頻 (PMM)", f"{stats['pmm']:.2f}")

            # 2. TF-IDF 全文 10 大核心關鍵字
            st.subheader("2. 全文 TF-IDF 核心關鍵詞 (Top 10)")
            df_tfidf = pd.DataFrame(stats["tfidf"], columns=["核心關鍵詞", "TF-IDF 權重分"])
            
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.bar_chart(df_tfidf.set_index("核心關鍵詞"))
            with col_t2:
                st.dataframe(df_tfidf, hide_index=True, use_container_width=True)

            # 3. 關鍵詞共現分析
            st.subheader(f"3. 「{target_word}」的前後常見共現詞 (Top 10 Co-occurrence)")
            df_co = pd.DataFrame(stats["co_occurrences"], columns=["共現詞彙", "共現次數"])
            
            if not df_co.empty:
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    st.bar_chart(df_co.set_index("共現詞彙"))
                with col_c2:
                    st.dataframe(df_co, hide_index=True, use_container_width=True)
            else:
                st.info(f"在文本中未找到「{target_word}」或周圍無顯著共現詞彙。")

            # 4. CSV 分析報表打包下載
            st.subheader("4. 匯出分析報表")
            
            # 建立多分頁 CSV DataBuffer
            df_summary = pd.DataFrame([{
                "指標項目": "總字數 (含標點)", "數值": stats['total_chars']},
                {"指標項目": "總詞數", "數值": stats['total_words']},
                {"指標項目": f"「{target_word}」出現次數", "數值": stats['target_count']},
                {"指標項目": "百萬詞頻 (PMM)", "數值": f"{stats['pmm']:.2f}"}
            ])
            
            # 將基礎統計、TF-IDF 與共現詞整合為單一 CSV 檔
            csv_buffer = io.StringIO()
            csv_buffer.write("=== 基礎敘述統計 ===\n")
            df_summary.to_csv(csv_buffer, index=False)
            csv_buffer.write("\n=== TF-IDF 核心關鍵詞 ===\n")
            df_tfidf.to_csv(csv_buffer, index=False)
            csv_buffer.write(f"\n=== 「{target_word}」共現詞統計 ===\n")
            df_co.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📊 下載完整文本分析 CSV 報表",
                data=csv_buffer.getvalue().encode('utf-8-sig'), # 使用 utf-8-sig 確保 Excel 打開中文無亂碼
                file_name=f"{os.path.splitext(st.session_state['file_name'])[0]}_analysis_report.csv",
                mime="text/csv"
            )
