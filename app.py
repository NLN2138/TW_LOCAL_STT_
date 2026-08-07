import os
import tempfile
import gc
import streamlit as st
from faster_whisper import WhisperModel

# 1. 頁面設定
st.set_page_config(
    page_title="台灣政治言談語音轉文字系統 (faster-whisper)",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 台灣政治言談語音轉文字 (ASR) 工具")
st.markdown("本系統採用 **faster-whisper (int8 量化)**，已專為 CPU 伺服器進行記憶體與速度優化。")

# 2. 載入模型 (使用 int8 量化，記憶體低於 2GB)
@st.cache_resource
def load_whisper_model():
    # 推薦選用 "medium" 或 "small"，在免費 CPU 上兼具精準度與速度
    model_size = "medium"
    
    # device="cpu", compute_type="int8" 是 CPU 提速與防崩潰的關鍵
    model = WhisperModel(
        model_size, 
        device="cpu", 
        compute_type="int8",
        cpu_threads=4  # 配合 Streamlit Cloud 的 CPU 核心數
    )
    return model

# 載入模型
try:
    with st.spinner("正在載入語音辨識模型 (僅初次載入需數秒)..."):
        model = load_whisper_model()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗：{e}")
    st.stop()

# 3. 檔案上傳介面
uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 開始辨識轉檔", type="primary"):
        # 寫入臨時檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("語音辨識中，請稍候..."):
            try:
                # 執行語音辨識
                # initial_prompt 傳入議會與台灣地方語境，大幅提升專有名詞辨識率
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=5,
                    language="zh",
                    initial_prompt="以下為台灣縣市議會質詢對話，包含議員、局處首長、預算案與地方建設討論。"
                )

                st.subheader("📝 辨識結果")
                
                full_text_list = []
                
                # 邊辨識邊以時間軸呈現
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    for segment in segments:
                        start = segment.start
                        end = segment.end
                        text = segment.text
                        
                        full_text_list.append(text)
                        
                        # 即時列出分段時間軸
                        st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {text}")

                full_text = "\n".join(full_text_list)

                # 提供完整逐字稿下載
                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"轉檔過程中發生錯誤：{e}")
            finally:
                # 記憶體回收
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                gc.collect()
