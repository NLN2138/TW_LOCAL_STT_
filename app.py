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
# 2. 模型載入機制 (安全型 CPU 設定)
# ==========================================
@st.cache_resource
def load_whisper_model():
    # 使用 "small" 或 "medium"，在免費雲端最安全
    model_size = "medium"
    
    # 移除 cpu_threads 硬綁定，防止觸發 Streamlit Cloud 資源限制被強制中斷
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8"
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
# 4. Prompt 設定與介面
# ==========================================
default_prompt = (
    "以下為中華民國台灣縣市議會與政壇討論質詢錄音。"
    "關鍵詞語：中華民國、台灣、臺灣、中國大陸、大陸、中國、兩岸關係、陸委會、中央政府、地方政府。"
    "請嚴格區分「台灣/臺灣」、「中華民國」、「中國大陸」、「中國」等不同政治實體稱謂。"
    "議事常見術語：總質詢、業務質詢、預算案、墊付案、三讀、附帶決議、請就座、時間暫停。"
    "常見單位與稱謂：議員、議長、縣長、市長、局長、處長、主計處、研考會、工務局、衛生局。"
)

with st.expander("⚙️ 高級設定：語意與政治名詞 Prompt 導引 (可自行微調)", expanded=True):
    st.info("💡 提示：此處已預先設定政治實體鎖定詞。可自行補充會議特定的『議員姓名』或『地方地名』。")
    custom_prompt = st.text_area(
        "語意導引 Prompt：",
        value=default_prompt,
        height=140
    )

uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

# ==========================================
# 5. 執行語音辨識 Logic (修正檔案存取機制)
# ==========================================
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 開始精準辨識轉檔", type="primary"):
        # 建立專用的臨時檔案以防止路徑衝突
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner("語音辨識中，請稍候 (CPU 運算中)..."):
                # 執行語音辨識
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=5,                      # 將 beam_size 調微降至 5 以大幅節省 CPU 計算與記憶體
                    language="zh",
                    temperature=0.0,
                    condition_on_previous_text=False, # 改為 False 可以預防長音訊解碼時的記憶體累積崩潰
                    initial_prompt=custom_prompt
                )

                st.subheader("📝 辨識結果")
                full_text_list = []
                
                # 使用串流/迭代方式逐段寫出，降低記憶體瞬間壓力
                time_placeholder = st.empty()
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    for segment in segments:
                        start = segment.start
                        end = segment.end
                        clean_text = refine_taiwan_terms(segment.text)
                        full_text_list.append(clean_text)
                        
                        st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {clean_text}")

                full_text = "\n".join(full_text_list)

                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"❌ 轉檔發生錯誤：{e}")
        finally:
            # 安全清理檔案與 GC
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            gc.collect()
