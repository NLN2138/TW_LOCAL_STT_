import os
import tempfile
import gc
import re
import streamlit as st
from faster_whisper import WhisperModel

# ==========================================
# 1. 頁面與 UI 基本配置
# ==========================================
st.set_page_config(
    page_title="台灣政治言談語音轉文字系統 (faster-whisper)",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 台灣政治言談語音轉文字 (ASR) 工具")
st.markdown("本系統採用 **faster-whisper (int8 CPU 量化優化)**，針對台灣縣市議會質詢、兩岸政治術語與在地地方語境進行強化。")

# ==========================================
# 2. 模型載入機制 (使用 cache 防止重複載入)
# ==========================================
@st.cache_resource
def load_whisper_model():
    # 使用 "medium" 尺寸模型，兼顧精準度與記憶體耗用
    model_size = "medium"
    
    # device="cpu", compute_type="int8" 是防崩潰與 CPU 提速的核心
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8",
        cpu_threads=4  # 配合 Streamlit Cloud 的免費 CPU 核心數
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
# 3. 後處理文本校正函數
# ==========================================
def refine_taiwan_terms(text: str) -> str:
    """
    針對 ASR 辨識結果進行臺灣政治專有名詞與用語的微調校正
    """
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

# ==========================================
# 4. Prompt 設定與檔案上傳介面
# ==========================================
# 預設的政治與議會專用 Context Prompt
default_prompt = (
    "以下為中華民國台灣縣市議會與政壇討論質詢錄音。"
    "關鍵詞語：中華民國、台灣、臺灣、中國大陸、大陸、中國、兩岸關係、陸委會、中央政府、地方政府。"
    "請嚴格區分「台灣/臺灣」、「中華民國」、「中國大陸」、「中國」等不同政治實體稱謂。"
    "議事常見術語：總質詢、業務質詢、預算案、墊付案、三讀、附帶決議、請就座、時間暫停。"
    "常見單位與稱謂：議員、議長、縣長、市長、局長、處長、主計處、研考會、工務局、衛生局。"
)

with st.expander("⚙️ 高級設定：語意與政治名詞 Prompt 導引 (可自行微調)", expanded=True):
    st.info("💡 提示：此處已預先設定「中華民國、台灣、中國大陸、中國」等政治實體鎖定詞。若有當次會議特定的『議員姓名』或『地方地名』，可直接補充在下方。")
    custom_prompt = st.text_area(
        "語意導引 Prompt：",
        value=default_prompt,
        height=140
    )

uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

# ==========================================
# 5. 執行語音辨識 logic (保持上個版本的最佳參數)
# ==========================================
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 開始精準辨識轉檔", type="primary"):
        tmp_path = None
        try:
            # 安全創建臨時檔案，避免檔名衝突與權限問題
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("語音辨識中，請稍候 (CPU 運算中，依音訊長度需數十秒至數分鐘)..."):
                # 完整保留上個版本的精準度參數組合
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=7,                      # Beam Search 設為 7，提高關鍵詞搜尋精準度
                    language="zh",
                    temperature=0.0,                  # 零隨機性，嚴格依據音訊與 Prompt 解碼
                    condition_on_previous_text=True,  # 鎖定前文脈絡，確保上下文專有名詞連貫
                    initial_prompt=custom_prompt      # 帶入政治與議會專屬 Prompt
                )

                st.subheader("📝 辨識結果")
                
                full_text_list = []
                
                # 分段顯示帶時間軸的逐字稿
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    for segment in segments:
                        start = segment.start
                        end = segment.end
                        
                        # 經過後處理校正的文字
                        clean_text = refine_taiwan_terms(segment.text)
                        full_text_list.append(clean_text)
                        
                        # 即時顯示時間軸格式 [MM:SS - MM:SS]
                        st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {clean_text}")

                full_text = "\n".join(full_text_list)

                # 下載按鈕
                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"❌ 轉檔過程中發生錯誤：{e}")
        finally:
            # 安全釋放臨時檔案與進行垃圾回收，避免記憶體洩漏引發 Oh no.
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            gc.collect()
