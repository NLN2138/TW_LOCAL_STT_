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

st.title("🎙️ 台灣政治言談語音轉文字與焦點分析工具")
st.markdown("本系統採用 **faster-whisper (int8 CPU 量化)**，支援客製化焦點關鍵詞辨識與自動化文本分析。")

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

def analyze_multiple_keywords(full_text: str, keywords: list, window_size: int = 5):
    """計算基礎統計、多個關注關鍵詞的 PMM/共現詞，以及全篇 TF-IDF"""
    clean_char_text = re.sub(r"[^\w]", "", full_text)
    total_chars = len(clean_char_text)
    
    # 將所有關注關鍵詞加入 jieba 防止被切碎
    for kw in keywords:
        if kw:
            jieba.add_word(kw)
    jieba.add_word("中華民國")
    jieba.add_word("中國大陸")
    
    words = [w for w in jieba.cut(full_text) if re.match(r"[\u4e00-\u9fa5a-zA-Z0-9]+", w) and len(w) > 1]
    total_words = len(words)
    
    if total_words == 0:
        return None

    # 1. 針對每個關注關鍵詞計算次數與 PMM
    kw_stats = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        count = full_text.count(kw)
        pmm = (count / total_words) * 1_000_000 if total_words > 0 else 0.0
        kw_stats.append({
            "關注關鍵詞": kw,
            "出現次數": count,
            "百萬詞頻 (PMM)": round(pmm, 2)
        })
    
    df_kw_summary = pd.DataFrame(kw_stats)

    # 2. TF-IDF 關鍵字提取
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
        "kw_summary": df_kw_summary,
        "tfidf": top_tfidf_words
    }

# ==========================================
# 4. 使用者自訂關注關鍵詞與 Prompt 介面
# ==========================================
st.subheader("🎯 第一步：設定分析焦點與關注關鍵詞")

user_keywords_input = st.text_input(
    "請輸入您這次想重點關注的詞彙或專有名詞（多個詞請用「逗號」隔開）：",
    value="台灣, 中華民國, 中國大陸, 質詢, 預算"
)

# 解析關鍵詞清單
user_keywords = [k.strip() for k in re.split(r"[,，]", user_keywords_input) if k.strip()]

# 自動根據使用者關鍵詞組裝動態 Prompt
dynamic_prompt = f"重點關注詞彙：{', '.join(user_keywords)}。" if user_keywords else ""
default_base_prompt = (
    "以下為中華民國台灣縣市議會與政壇討論質詢錄音。"
    "請嚴格區分「台灣/臺灣」、「中華民國」、「中國大陸」、「中國」等不同政治實體稱謂。"
    "議事常見術語：總質詢、業務質詢、預算案、墊付案、三讀、附帶決議、請就座、時間暫停。"
    "常見單位與稱謂：議員、議長、縣長、市長、局長、處長、主計處、研考會、工務局、衛生局。"
)

full_custom_prompt = dynamic_prompt + default_base_prompt

with st.expander("⚙️ 檢視系統自動生成的模型語意 Prompt (會將您的關鍵詞優先注入)", expanded=False):
    st.text_area("實際帶入 Whisper 的 Prompt：", value=full_custom_prompt, height=120)

# ==========================================
# 5. 上傳提示卡片與檔案上傳
# ==========================================
st.info("""
⏱️ **音訊上傳限制與處理時間說明 (純 CPU 伺服器環境)：**
* **建議音訊長度：** 請控制在 **10 分鐘以內**（若檔期過長，建議分割後分段上傳）。
* **長度上限警示：** 超過 **20 分鐘** 的音訊有機率觸發伺服器 Timeout 逾時連線中斷。
* **預估處理時間：**
  * ⏱️ **1 ~ 3 分鐘音訊：** 約需 30 秒 ~ 1 分鐘
  * ⏱️ **5 ~ 10 分鐘音訊：** 約需 2 ~ 4 分鐘
""")

uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

# ==========================================
# 6. 執行語音辨識與文本分析
# ==========================================
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 開始精準辨識與焦點分析", type="primary"):
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
                    initial_prompt=full_custom_prompt # 注入含使用者關鍵詞的 Prompt
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
                st.session_state["user_keywords"] = user_keywords

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
# 7. 自動焦點分析與 CSV 導出模組
# ==========================================
if "full_text" in st.session_state and st.session_state["full_text"]:
    st.markdown("---")
    st.header("📊 您關注焦點的量化文本分析")
    
    full_text = st.session_state["full_text"]
    keywords = st.session_state.get("user_keywords", [])

    with st.spinner("正在計算焦點詞頻與 TF-IDF 關鍵字..."):
        stats = analyze_multiple_keywords(full_text, keywords)

    if stats:
        # 1. 全文基礎統計
        m1, m2 = st.columns(2)
        m1.metric("總字數 (含標點)", f"{stats['total_chars']} 字")
        m2.metric("總詞數 (Segs)", f"{stats['total_words']} 詞")

        # 2. 自訂關注詞頻統計看板與圖表
        st.subheader("1. 自訂關注詞出現頻率與百萬詞頻 (PMM)")
        df_kw = stats["kw_summary"]
        
        if not df_kw.empty:
            col_k1, col_k2 = st.columns([2, 1])
            with col_k1:
                st.bar_chart(df_kw.set_index("關注關鍵詞")["百萬詞頻 (PMM)"])
            with col_k2:
                st.dataframe(df_kw, hide_index=True, use_container_width=True)
        else:
            st.info("未輸入任何關注關鍵詞。")

        # 3. TF-IDF 全文 10 大核心關鍵字
        st.subheader("2. 全文自動提取 TF-IDF 核心主題詞 (Top 10)")
        df_tfidf = pd.DataFrame(stats["tfidf"], columns=["核心關鍵詞", "TF-IDF 權重分"])
        
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.bar_chart(df_tfidf.set_index("核心關鍵詞"))
        with col_t2:
            st.dataframe(df_tfidf, hide_index=True, use_container_width=True)

        # 4. 打包 CSV 匯出
        st.subheader("3. 匯出焦點分析報表")
        
        csv_buffer = io.StringIO()
        csv_buffer.write("=== 基礎語料統計 ===\n")
        pd.DataFrame([{"總字數": stats['total_chars'], "總詞數": stats['total_words']}]).to_csv(csv_buffer, index=False)
        csv_buffer.write("\n=== 使用者自訂關注詞統計 ===\n")
        df_kw.to_csv(csv_buffer, index=False)
        csv_buffer.write("\n=== TF-IDF 全文主題詞 ===\n")
        df_tfidf.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📊 下載關注焦點分析 CSV 報表",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name=f"{os.path.splitext(st.session_state['file_name'])[0]}_focus_report.csv",
            mime="text/csv"
        )
