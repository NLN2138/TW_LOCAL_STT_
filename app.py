import os
import tempfile
import gc
import torch
import streamlit as st
from transformers import pipeline

# 1. 頁面設定
st.set_page_config(
    page_title="台灣政治言談語音轉文字系統 (聯發科 Breeze-ASR)",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ 台灣政治言談語音轉文字 (ASR) 工具")
st.markdown("本工具採用 **MediaTek Research Breeze-ASR** 模型，專為台灣口音、中台語夾雜與地方語境優化。")

# 2. 模型載入機制 (使用 cache 防止重複載入)
@st.cache_resource
def load_asr_pipeline():
    model_id = "MediaTek-Research/Breeze-ASR-25"
    
    # 判斷硬體環境
    is_cuda = torch.cuda.is_available()
    device = "cuda:0" if is_cuda else "cpu"
    torch_dtype = torch.float16 if is_cuda else torch.float32

    st.info(f"正在載入模型 `{model_id}` (運行平台: {device.upper()})...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,      # 音訊分區長度
        stride_length_s=(4, 2),  # 重疊滑動視窗
    )
    return pipe

# 載入模型
try:
    asr_pipe = load_asr_pipeline()
    st.success("模型載入成功！")
except Exception as e:
    st.error(f"模型載入失敗，可能因記憶體不足被系統強制終止：{e}")
    st.stop()

# 3. 檔案上傳介面
uploaded_file = st.file_uploader("請上傳議會開會 MP3 音訊檔", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    
    # 上傳檔案預警：免費 CPU 建議控制在 10 分鐘以內
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 15:
        st.warning(f"⚠️ 上傳檔案較大 ({file_size_mb:.1f} MB)，在免費 CPU 伺服器上運算可能需要較長時間或遭遇記憶體限制。")

    if st.button("🚀 開始辨識轉檔", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("語音辨識中（CPU 算力較慢，請耐心等候）..."):
            try:
                # 執行語音辨識
                result = asr_pipe(
                    tmp_path, 
                    return_timestamps=True,
                    batch_size=1, # 控制 batch 大小以極致節省 RAM
                    generate_kwargs={
                        "language": "zh", 
                        "task": "transcribe",
                        # 加入台灣議會 Context 提示詞，提升專有名詞辨識力
                        "prompt": "以下為台灣縣市議會質詢對話，包含議員、局處首長、地方建設與預算案討論。"
                    }
                )
                
                full_text = result.get("text", "")
                chunks = result.get("chunks", [])

                st.subheader("📝 辨識結果")
                
                # 分段顯示時間軸
                with st.expander("檢視逐字稿時間軸", expanded=True):
                    if chunks:
                        for chunk in chunks:
                            timestamp = chunk.get("timestamp")
                            text = chunk.get("text", "")
                            if timestamp and len(timestamp) == 2 and timestamp[0] is not None and timestamp[1] is not None:
                                start, end = timestamp
                                st.markdown(f"**[{int(start//60):02d}:{int(start%60):02d} - {int(end//60):02d}:{int(end%60):02d}]** {text}")
                            else:
                                st.markdown(f"{text}")
                    else:
                        st.write(full_text)

                # 下載按鈕
                st.download_button(
                    label="💾 下載完整 txt 逐字稿",
                    data=full_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"轉檔過程中發生錯誤（可能是記憶體超限）：{e}")
            finally:
                # 清除檔案與強制進行垃圾回收 (GC) 釋放記憶體
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                gc.collect()
