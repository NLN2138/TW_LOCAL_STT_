import os
import tempfile
import torch
import streamlit as st
from transformers import pipeline

# 1. 頁面設定
st.set_page_config(
    page_title="台灣議會語音轉文字系統 (聯發科 Breeze-ASR)",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 台灣縣市議會語音轉文字 (ASR) 工具")
st.markdown("本工具採用 **MediaTek Research Breeze-ASR** 模型，專為台灣口音、中台語夾雜與地方語境優化。")

# 2. 模型載入機制 (使用 cache 防止重複載入)
@st.cache_resource
def load_asr_pipeline():
    # 選擇聯發科 ASR 模型
    model_id = "MediaTek-Research/Breeze-ASR-25"
    
    # 判斷硬體環境 (GPU 或 CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    st.info(f"正在載入模型 `{model_id}` (運行平台: {device.upper()})...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        # 自動處理長音訊切割
        chunk_length_s=30,
        stride_length_s=(4, 2),
    )
    return pipe

# 載入模型
try:
    asr_pipe = load_asr_pipeline()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，請檢查網路或硬體配置：{e}")
    st.stop()

# 3. 檔案上傳介面
uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 播放音訊 preview
    st.audio(uploaded_file, format="audio/mp3")
    
    # 將上傳的檔案存為臨時檔供 pipeline 讀取
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if st.button("🚀 開始辨識轉檔", type="primary"):
        with st.spinner("語音辨識中，請稍候（較長的音訊可能需要數分鐘）..."):
            try:
                # 執行語音辨識 (產生 timestamp 方便核對議會紀錄)
                result = asr_pipe(
                    tmp_path, 
                    return_timestamps=True,
                    generate_kwargs={"language": "zh", "task": "transcribe"}
                )
                
                full_text = result["text"]
                chunks = result.get("chunks", [])

                st.subheader("📝 辨識結果")
                
                # 分段顯示帶時間軸的文字
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    for chunk in chunks:
                        start, end = chunk["timestamp"]
                        text = chunk["text"]
                        st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {text}")

                # 整體文字下載區
                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"轉檔過程中發生錯誤：{e}")
            finally:
                # 清理臨時檔案
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
