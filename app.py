import os
import tempfile
import gc
import re
import io
from collections import Counter
import streamlit as st
import pandas as pd
from faster_whisper import WhisperModel

# ==========================================
# 1. 頁面配置與自訂美化 CSS
# ==========================================
st.set_page_config(
    page_title="台灣政治言談 ASR 與語料庫分析系統",
    page_icon="🎙️",
    layout="wide"
)

# 透過自訂 CSS 調優字體、間距與標題層級（標題顏色為白色）
st.markdown("""
    <style>
    /* 調整全局容器邊界與背景 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* 大標題樣式 - 純白 */
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-bottom: 0.2rem !important;
    }
    /* 副標題樣式 - 淺灰白 */
    .sub-title {
        font-size: 1.05rem !important;
        color: #E2E8F0 !important;
        margin-bottom: 1.8rem !important;
    }

    /* 區塊標題（Section Titles）樣式 - 純白 */
    .section-header {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        border-left: 4px solid #3B82F6;
        padding-left: 0.6rem;
        margin-top: 1.2rem !important;
        margin-bottom: 1.0rem !important;
    }

    /* 調整 Expander 區塊標題文字顏色為亮白 */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #F8FAFC !important;
    }
    
    /* 微調分隔線間距 */
    hr {
        margin-top: 1.8rem !important;
        margin-bottom: 1.8rem !important;
        border-color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 頁頭主標題
st.markdown('<p class="main-title">🎙️ 台灣政治言談語音轉文字與語料庫分析系統 v1.0 （雛形）</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">專為語言學與政治言談研究設計 | 整合 faster-whisper ASR 與客觀語料庫敘述統計 (PMM, KWIC, TF-IDF)</p>', unsafe_allow_html=True)

try:
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    st.error("❌ 未安裝必要套件，請於 requirements.txt 加入 jieba 與 scikit-learn。")

DEFAULT_STOPWORDS = set([
    "這個", "那個", "我們", "你們", "他們", "因為", "所以", "但是", "然後", "如果",
    "進行", "表示", "認為", "相當", "非常", "關於", "對於", "可以", "可能", "應該"
])

DEFAULT_KEYWORDS_LIST = ["測試", "這", "語音", "分鐘"]

# ==========================================
# 2. 模型載入機制
# ==========================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=4)

try:
    with st.spinner("正在初始化語音辨識模型，請稍候..."):
        model = load_whisper_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# ==========================================
# 3. 核心語言學處理函數
# ==========================================
def refine_taiwan_terms(text: str) -> str:
    """
    優化政治與在地術語：
    1. 統一文字異體字（優先將「臺灣」統一替換為「台灣」）
    2. 優先替換長字詞（如「中華民國」、「中國大陸」），避免被短字詞拆解
    """
    if not text:
        return ""
    
    # 步驟 1: 異體字統一標準化
    text = text.replace("臺灣", "台灣")
    
    # 步驟 2: 長詞優先正則與對應（確保中華民國、中國大陸不被誤校正）
    replacements = [
        (r"中華民國", "中華民國"),
        (r"中國大陸", "中國大陸"),
        (r"大陸委員會|陸委會", "陸委會"),
        (r"海峽交流基金會|海基會", "海基會"),
        (r"台灣", "台灣"),
    ]
    
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
        
    return text

def generate_srt(segments) -> str:
    srt_output = []
    for i, seg in enumerate(segments, start=1):
        s_min, s_sec = divmod(seg['start'], 60)
        s_hr, s_min = divmod(s_min, 60)
        e_min, e_sec = divmod(seg['end'], 60)
        e_hr, e_min = divmod(e_min, 60)
        
        timestamp = f"{int(s_hr):02d}:{int(s_min):02d}:{s_sec:06.3f}".replace('.', ',') + " --> " + f"{int(e_hr):02d}:{int(e_min):02d}:{e_sec:06.3f}".replace('.', ',')
        srt_output.append(f"{i}\n{timestamp}\n{seg['text']}\n")
    return "\n".join(srt_output)

def analyze_corpus_descriptive(full_text: str, keywords: list, segments: list, window_size: int = 5):
    clean_char_text = re.sub(r"[^\w]", "", full_text)
    total_chars = len(clean_char_text)
    
    # 動態註冊關注重點詞與政治專有名詞，避免 jieba 誤切
    for kw in keywords:
        if kw:
            jieba.add_word(kw)
    jieba.add_word("中華民國")
    jieba.add_word("中國大陸")
    jieba.add_word("台灣")
    jieba.add_word("陸委會")
    jieba.add_word("海基會")
    
    raw_tokens = [w.strip() for w in jieba.cut(full_text) if w.strip() and re.match(r"[\u4e00-\u9fa5a-zA-Z0-9]+", w)]
    total_tokens = len(raw_tokens)
    
    if total_tokens == 0:
        return None

    token_counts = Counter(raw_tokens)
    kw_stats = []
    kwic_results = []

    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        count = token_counts.get(kw, 0)
        pmm = (count / total_tokens) * 1_000_000 if total_tokens > 0 else 0.0
        
        kw_stats.append({
            "關注關鍵詞": kw,
            "出現次數 (Raw Frequency)": count,
            "百萬詞頻 (PMM)": round(pmm, 2)
        })

        for i, token in enumerate(raw_tokens):
            if token == kw:
                left_context = "".join(raw_tokens[max(0, i - window_size):i])
                right_context = "".join(raw_tokens[i + 1:min(len(raw_tokens), i + window_size + 1)])
                kwic_results.append({
                    "關鍵詞": kw,
                    "左語境 (Left Context)": left_context,
                    "焦點詞 (Node)": token,
                    "右語境 (Right Context)": right_context
                })

    sentences = [s.strip() for s in re.split(r"[。！？\n]", full_text) if len(s.strip()) > 5]
    top_tfidf_words = []
    if sentences:
        corpus = [" ".join([w for w in jieba.cut(s) if w not in DEFAULT_STOPWORDS and len(w) > 1]) for s in sentences]
        corpus = [doc for doc in corpus if doc.strip()]
        if corpus:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            sums = tfidf_matrix.sum(axis=0)
            
            data = []
            for col, cost in enumerate(sums.A[0]):
                word = feature_names[col]
                if not word.isdigit():
                    data.append((word, round(cost, 3)))
            data.sort(key=lambda x: x[1], reverse=True)
            top_tfidf_words = data[:10]

    total_duration = segments[-1]['end'] if segments else 0
    wpm = (total_chars / (total_duration / 60)) if total_duration > 0 else 0

    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_duration": round(total_duration, 1),
        "wpm": round(wpm, 1),
        "kw_summary": pd.DataFrame(kw_stats),
        "kwic": pd.DataFrame(kwic_results),
        "tfidf": pd.DataFrame(top_tfidf_words, columns=["主題詞", "TF-IDF 權重分"])
    }

# ==========================================
# 4. 第一階段：任務參數與音訊配置 (卡片化外框)
# ==========================================
st.markdown('<p class="section-header">第一步：設定關注關鍵詞與音訊來源</p>', unsafe_allow_html=True)

with st.container(border=True):
    col_kw, col_up = st.columns([1, 1], gap="medium")
    
    with col_kw:
        st.markdown("**🎯 關注焦點詞設定**")
        user_keywords_input = st.text_input(
            "請輸入研究關注詞彙（以逗號分隔）：",
            placeholder="例如：建設, 交通, 民生",
            value=""
        )
        if user_keywords_input.strip():
            current_keywords = [k.strip() for k in re.split(r"[,，]", user_keywords_input) if k.strip()]
            st.caption(f"📌 當前採用關鍵詞：`{', '.join(current_keywords)}`")
        else:
            current_keywords = DEFAULT_KEYWORDS_LIST
            st.caption(f"💡 未填寫則啟用預設關鍵詞：`{', '.join(DEFAULT_KEYWORDS_LIST)}`")

    with col_up:
        st.markdown("**📁 語音檔案選擇**")
        uploaded_file = st.file_uploader("上傳錄音檔 (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"], label_visibility="collapsed")
        
        audio_source_path = None
        file_display_name = ""

        if uploaded_file is not None:
            file_display_name = uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_source_path = tmp_file.name
            st.audio(uploaded_file)
        else:
            default_audio_file = "test_audio.mp3"
            if os.path.exists(default_audio_file):
                audio_source_path = default_audio_file
                file_display_name = "test_audio.mp3 (預設測試檔)"
                st.caption("🎵 未上傳檔案，自動使用專案預設檔 `test_audio.mp3`")
                st.audio(default_audio_file)
            else:
                st.error("⚠️ 尚未上傳檔案，且根目錄未找到 `test_audio.mp3`。")

    # 執行按鈕
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    btn_submit = st.button("🚀 開始辨識與語料庫建置", type="primary", use_container_width=True)

# 組合 Prompt
dynamic_prompt = f"重點關注詞彙：{', '.join(current_keywords)}。"
default_base_prompt = (
    "以下為中華民國台灣縣市議會與政壇討論質詢錄音。"
    "請嚴格區分「台灣/臺灣」、「中華民國」、「中國大陸」、「中國」等不同政治實體稱謂。"
    "常見術語：總質詢、業務質詢、預算案、墊付案、三讀、附帶決議。"
)
full_custom_prompt = dynamic_prompt + default_base_prompt

# ==========================================
# 5. 語音辨識執行與結果處理
# ==========================================
if audio_source_path is not None and btn_submit:
    with st.spinner(f"正在辨識音檔 `{file_display_name}` 中，請稍候..."):
        try:
            segments_raw, info = model.transcribe(
                audio_source_path,
                beam_size=7,
                language="zh",
                temperature=0.0,
                condition_on_previous_text=True,
                initial_prompt=full_custom_prompt
            )

            parsed_segments = []
            full_text_list = []

            for seg in segments_raw:
                clean_txt = refine_taiwan_terms(seg.text)
                full_text_list.append(clean_txt)
                parsed_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": clean_txt
                })

            st.session_state["full_text"] = "\n".join(full_text_list)
            st.session_state["segments"] = parsed_segments
            st.session_state["file_name"] = file_display_name
            st.success("✅ 辨識與轉檔完成！")

        except Exception as e:
            st.error(f"❌ 轉檔錯誤：{e}")
        finally:
            if uploaded_file is not None and audio_source_path and os.path.exists(audio_source_path):
                os.remove(audio_source_path)
            gc.collect()

# ==========================================
# 6. 第二階段：成果展示與語料庫分析 (Tab 頁籤分頁美化)
# ==========================================
if "full_text" in st.session_state:
    st.markdown('<p class="section-header">第二步：逐字稿檢視與語料庫分析</p>', unsafe_allow_html=True)
    
    full_text = st.session_state["full_text"]
    segments = st.session_state["segments"]

    # 使用 Streamlit Tabs 將「逐字稿」與「統計分析」優雅分開
    tab_transcript, tab_analysis = st.tabs(["📝 逐字稿與下載", "📊 語料庫敘述統計分析"])

    # --- TAB 1: 逐字稿區 ---
    with tab_transcript:
        with st.container(border=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "💾 下載 TXT 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(st.session_state['file_name'])[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_d2:
                srt_data = generate_srt(segments)
                st.download_button(
                    "🎬 下載 SRT 字幕檔 (含時間軸)",
                    data=srt_data,
                    file_name=f"{os.path.splitext(st.session_state['file_name'])[0]}.srt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with st.expander("檢視完整分段時間軸逐字稿", expanded=True):
            for seg in segments:
                s_m, s_s = int(seg['start']//60), int(seg['start']%60)
                e_m, e_s = int(seg['end']//60), int(seg['end']%60)
                st.markdown(f"**`[{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}]`** {seg['text']}")

    # --- TAB 2: 語料庫分析區 ---
    with tab_analysis:
        window_size = st.slider("調整 KWIC 語境視窗長度 (前後字數)：", min_value=3, max_value=15, value=5)
        stats = analyze_corpus_descriptive(full_text, current_keywords, segments, window_size)

        if stats:
            # 1. 語料庫基本數據卡片 (看板)
            st.markdown("##### 1. 語料庫基本屬性與語速")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總字數 (Characters)", f"{stats['total_chars']} 字")
            m2.metric("總詞數 (Tokens)", f"{stats['total_tokens']} 詞")
            m3.metric("錄音總時長", f"{stats['total_duration']} 秒")
            m4.metric("平均語速 (CPM)", f"{stats['wpm']} 字/分")

            st.markdown("---")

            # 2. PMM 統計與圖表
            st.markdown("##### 2. 關注詞出現頻率與百萬詞頻 (PMM)")
            if not stats["kw_summary"].empty:
                k1, k2 = st.columns([3, 2], gap="medium")
                with k1:
                    st.dataframe(stats["kw_summary"], hide_index=True, use_container_width=True)
                with k2:
                    st.bar_chart(stats["kw_summary"].set_index("關注關鍵詞")["百萬詞頻 (PMM)"])

            st.markdown("---")

            # 3. KWIC 語境檢索
            st.markdown("##### 3. KWIC 關鍵詞語境檢索 (Key Word in Context)")
            if not stats["kwic"].empty:
                st.dataframe(stats["kwic"], use_container_width=True)
            else:
                st.info("未在語料庫中找到關注關鍵詞的上下文。")

            st.markdown("---")

            # 4. TF-IDF
            st.markdown("##### 4. 實詞 TF-IDF 權重排名 (已過濾虛詞/停用詞)")
            if not stats["tfidf"].empty:
                t1, t2 = st.columns([3, 2], gap="medium")
                with t1:
                    st.dataframe(stats["tfidf"], hide_index=True, use_container_width=True)
                with t2:
                    st.bar_chart(stats["tfidf"].set_index("主題詞"))

            st.markdown("---")

            # 5. 打包匯出
            st.markdown("##### 5. 匯出研究報表")
            csv_buffer = io.StringIO()
            csv_buffer.write("=== 語料庫基本屬性 ===\n")
            pd.DataFrame([{
                "總字數": stats['total_chars'], 
                "總詞數": stats['total_tokens'], 
                "錄音總時長(秒)": stats['total_duration'],
                "平均語速(字/分)": stats['wpm']
            }]).to_csv(csv_buffer, index=False)
            
            csv_buffer.write("\n=== 關注詞頻與 PMM 敘述統計 ===\n")
            stats["kw_summary"].to_csv(csv_buffer, index=False)

            csv_buffer.write("\n=== KWIC 語境檢索表 ===\n")
            stats["kwic"].to_csv(csv_buffer, index=False)
            
            csv_buffer.write("\n=== 實詞 TF-IDF 權重表 ===\n")
            stats["tfidf"].to_csv(csv_buffer, index=False)

            st.download_button(
                "📊 下載完整語言學敘述統計報告 (CSV)",
                data=csv_buffer.getvalue().encode('utf-8-sig'),
                file_name=f"{os.path.splitext(st.session_state['file_name'])[0]}_descriptive_report.csv",
                mime="text/csv",
                type="primary"
            )

# ==========================================
# 7. 頁尾免責聲明 (弱化處理，避免搶視覺焦點)
# ==========================================
st.markdown("---")
with st.expander("⚠️ 系統與學術免責聲明 (Disclaimer)"):
    st.caption("""
    1. **ASR 辨識精準度限制**：本系統採用自動語音辨識技術 (Faster-Whisper/ASR)，辨識結果受錄音品質、重疊發音、地方口音與背景雜音影響。逐字稿內容僅供學術研究與初步參考，正式研究引用前請務必進行人工校對。
    2. **數據統計說明**：本系統提供之詞頻、PMM (Per Million Marks) 及 TF-IDF 為基於自動斷詞工具 (Jieba) 之客觀敘述性統計結果，不包含推論統計或價值判斷。
    3. **個人資料與著作權**：使用者上傳之音訊檔應確保符合個人資料保護法及相關著作權規範，本系統伺服器不會永久儲存使用者上傳之原始影音檔案。
    """)
