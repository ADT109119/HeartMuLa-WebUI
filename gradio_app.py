"""
HeartLib Gradio Interface
A web-based interface for music generation using HeartMuLaGenPipeline.
"""

import os
import sys
import gc
import time
import torch
import gradio as gr
from datetime import datetime
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from heartlib import HeartMuLaGenPipeline


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_PATH = "./ckpt"
OUTPUT_DIR = "./output"
FFMPEG_DIR = "./ffmpeg/bin"  # Local ffmpeg directory


def setup_ffmpeg_path():
    """
    Add local ffmpeg bin directory to PATH if it exists.
    This allows bundling ffmpeg with the application without system-wide installation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin_path = os.path.join(script_dir, FFMPEG_DIR)
    
    if os.path.exists(ffmpeg_bin_path):
        # Add to PATH (prepend to take priority over system ffmpeg)
        os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ.get("PATH", "")
        print(f"✅ FFmpeg 路徑已設定: {ffmpeg_bin_path}")
        return True
    else:
        print(f"⚠️ FFmpeg 目錄不存在: {ffmpeg_bin_path}")
        print("   請將 ffmpeg 的 bin 資料夾放置於 ./ffmpeg/bin/")
        return False


# Initialize ffmpeg path on module load
setup_ffmpeg_path()


def get_available_devices():
    """
    Detect available compute devices (CPU and CUDA GPUs).
    
    Returns:
        list: List of device strings like ['cpu', 'cuda:0', 'cuda:1', ...]
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i} ({gpu_name})")
    return devices


def parse_device_string(device_str: str) -> str:
    """
    Parse device string to extract the device identifier.
    
    Args:
        device_str: Device string like 'cuda:0 (NVIDIA RTX 4090)'
    
    Returns:
        str: Clean device string like 'cuda:0'
    """
    if "(" in device_str:
        return device_str.split(" (")[0]
    return device_str


# ============================================================================
# Throttled Progress Tracking
# ============================================================================

class ThrottledTqdmProgress:
    """
    A context manager that patches tqdm to update Gradio progress at reduced frequency.
    This avoids the overhead of updating on every iteration while still showing progress.
    """
    
    def __init__(self, progress_callback, update_interval: int = 50):
        """
        Args:
            progress_callback: Gradio progress function
            update_interval: Only update progress every N iterations
        """
        self.progress = progress_callback
        self.update_interval = update_interval
        self._original_tqdm = None
    
    def __enter__(self):
        import tqdm as tqdm_module
        self._original_tqdm = tqdm_module.tqdm
        
        progress = self.progress
        update_interval = self.update_interval
        
        class ThrottledTqdm(self._original_tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._gradio_update_counter = 0
                self._last_progress_update = 0
            
            def update(self, n=1):
                result = super().update(n)
                self._gradio_update_counter += n
                
                # Determine effective update interval based on total steps
                # Only throttle if total > 100 steps
                effective_interval = update_interval if (self.total and self.total > 100) else 1
                
                # Update Gradio progress
                if self.total and self._gradio_update_counter >= effective_interval:
                    current_progress = self.n / self.total
                    # Calculate speed
                    elapsed = self.format_dict.get('elapsed', 0)
                    rate = self.n / elapsed if elapsed > 0 else 0
                    
                    progress(
                        current_progress,
                        desc=f"生成中... {self.n}/{self.total} ({rate:.1f} it/s)"
                    )
                    self._gradio_update_counter = 0
                
                return result
        
        # Patch tqdm globally
        tqdm_module.tqdm = ThrottledTqdm
        
        # Also patch the tqdm import in heartlib if it's already imported
        try:
            from heartlib.pipelines import music_generation
            music_generation.tqdm = ThrottledTqdm
        except:
            pass
        
        return self
    
    def __exit__(self, *args):
        import tqdm as tqdm_module
        tqdm_module.tqdm = self._original_tqdm
        
        # Restore original in heartlib
        try:
            from heartlib.pipelines import music_generation
            music_generation.tqdm = self._original_tqdm
        except:
            pass


# ============================================================================
# Music Generation Function
# ============================================================================

def generate_music(
    lyrics: str,
    tags: str,
    max_audio_length_sec: int,
    topk: int,
    temperature: float,
    cfg_scale: float,
    model_version: str,
    mula_device: str,
    codec_device: str,
    mula_dtype: str,
    codec_dtype: str,
    lazy_load: bool,
    quantize_4bit: bool,
    progress=gr.Progress(),
):
    """
    Generate music using HeartMuLaGenPipeline with progress tracking.
    
    Args:
        lyrics: Song lyrics text
        tags: Comma-separated music style tags
        max_audio_length_sec: Maximum audio length in seconds
        topk: Top-K sampling parameter
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        model_version: Model version (e.g., "3B")
        mula_device: Device for MuLa model
        codec_device: Device for codec model
        mula_dtype: Data type for MuLa model
        codec_dtype: Data type for codec model
        lazy_load: Whether to use lazy loading
        quantize_4bit: Whether to use 4-bit quantization
        progress: Gradio progress tracker
    
    Returns:
        tuple: (audio_path, log_output)
    """
    log_lines = []
    
    def log(msg):
        """Add message to log and print to console."""
        log_lines.append(msg)
        print(msg)
    
    # Validate inputs
    if not lyrics.strip():
        return None, "❌ 錯誤：請輸入歌詞"
    
    if not tags.strip():
        return None, "❌ 錯誤：請輸入音樂標籤"
    
    output_path = None
    
    try:
        # Parse devices
        mula_dev = torch.device(parse_device_string(mula_device))
        codec_dev = torch.device(parse_device_string(codec_device))
        
        # Parse dtypes
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        mula_dt = dtype_map.get(mula_dtype, torch.bfloat16)
        codec_dt = dtype_map.get(codec_dtype, torch.float32)
        
        log(f"🎵 開始音樂生成...")
        log(f"📝 歌詞長度: {len(lyrics)} 字符")
        log(f"🏷️  標籤: {tags}")
        log(f"⚙️  參數: 長度={max_audio_length_sec}s, topk={topk}, temp={temperature}, cfg={cfg_scale}")
        log(f"🖥️  裝置: MuLa={mula_dev}, Codec={codec_dev}")
        log(f"📊 類型: MuLa={mula_dtype}, Codec={codec_dtype}")
        log(f" Lazy Load: {'啟用' if lazy_load else '停用'}")
        log(f"🤏 4-bit 量化: {'啟用' if quantize_4bit else '停用'}")
        log("-" * 50)
        
        # Configure 4-bit quantization
        bnb_config = None
        if quantize_4bit:
            if not torch.cuda.is_available() or mula_dev.type != "cuda":
                log("⚠️ 警告: 4-bit 量化需要 CUDA 裝置。已忽略此設定。")
            elif BitsAndBytesConfig is None:
                log("⚠️ 警告: 未安裝 transformers 或 bitsandbytes，無法使用 4-bit 量化。")
            else:
                quant_type = "nf4"
                try:
                    # Check for fp4 support (Compute Capability >= 8.9 for Ada Lovelace)
                    major, _ = torch.cuda.get_device_capability(mula_dev)
                    if major >= 10: 
                        quant_type = "fp4"
                except:
                    pass

                log(f"⚙️  啟用 4-bit 量化 (type: {quant_type})")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_type,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}.mp3")
        
        # Load pipeline
        progress(0.0, desc="載入模型中...")
        log("🔄 載入模型中...")
        
        pipe = HeartMuLaGenPipeline.from_pretrained(
            DEFAULT_MODEL_PATH,
            device={
                "mula": mula_dev,
                "codec": codec_dev,
            },
            dtype={
                "mula": mula_dt,
                "codec": codec_dt,
            },
            version=model_version,
            lazy_load=lazy_load,
            bnb_config=bnb_config,
        )
        log("✅ 模型載入完成")
        
        # Convert seconds to milliseconds
        max_audio_length_ms = max_audio_length_sec * 1000
        max_frames = max_audio_length_ms // 80  # Each frame is 80ms
        
        # Generate music with time tracking
        progress(0.1, desc="生成音樂中...")
        log("🎼 生成音樂中...")
        log(f"📊 最大幀數: {max_frames} frames")
        
        start_time = time.time()
        
        # Use throttled progress to avoid overhead (updates every 50 iterations)
        with ThrottledTqdmProgress(progress, update_interval=50):
            with torch.no_grad():
                pipe(
                    {
                        "lyrics": lyrics,
                        "tags": tags,
                    },
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=output_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_speed = max_frames / elapsed_time if elapsed_time > 0 else 0
        
        progress(1.0, desc="完成！")
        log("-" * 50)
        log(f"✅ 生成完成！")
        log(f"⏱️  總耗時: {elapsed_time:.2f} 秒")
        log(f"⚡ 平均速度: {avg_speed:.2f} frames/s")
        log(f"📁 輸出檔案: {output_path}")
        
        # Cleanup
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return output_path, "\n".join(log_lines)
        
    except Exception as e:
        error_msg = f"❌ 生成失敗: {str(e)}"
        log(error_msg)
        import traceback
        log(traceback.format_exc())
        return None, "\n".join(log_lines)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Get available devices
    devices = get_available_devices()
    default_device = devices[1] if len(devices) > 1 else devices[0]  # Prefer first GPU
    
    # Sample lyrics
    sample_lyrics = """[Intro]

[Verse]
陽光灑落在窗台上
新的一天開始閃亮
心中有夢想要飛翔
讓音樂帶我去遠方

[Chorus]
唱出心中的旋律
讓快樂傳遞
每一個音符都是奇蹟
這就是我的音樂之旅

[Outro]
(漸漸淡出...)"""

    sample_tags = "pop,piano,upbeat,cheerful,female voice"
    
    # Build interface
    with gr.Blocks(
        title="HeartLib Music Generation",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 1rem; }
        .log-box textarea { 
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important; 
            font-size: 13px !important;
            line-height: 1.5 !important;
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🎵 HeartLib Music Generation
            使用 AI 根據歌詞和風格標籤生成音樂
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📝 輸入")
                
                lyrics_input = gr.Textbox(
                    label="歌詞",
                    placeholder="輸入歌詞，可使用 [Intro], [Verse], [Chorus] 等標記...",
                    lines=12,
                    value=sample_lyrics,
                )
                
                tags_input = gr.Textbox(
                    label="音樂標籤",
                    placeholder="例如: pop, rock, piano, upbeat, male voice",
                    value=sample_tags,
                )
                
                gr.Markdown("### ⚙️ 參數設定")
                
                with gr.Row():
                    max_length = gr.Slider(
                        label="音訊長度 (秒)",
                        minimum=30,
                        maximum=360,
                        value=240,
                        step=10,
                    )
                    
                    topk = gr.Slider(
                        label="Top-K",
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                    )
                
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )
                    
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                    )
                
                gr.Markdown("### 🖥️ 裝置設定")
                
                model_version = gr.Dropdown(
                    label="模型版本",
                    choices=["3B"],
                    value="3B",
                )
                
                with gr.Row():
                    mula_device = gr.Dropdown(
                        label="MuLa 裝置",
                        choices=devices,
                        value=default_device,
                    )
                    
                    codec_device = gr.Dropdown(
                        label="Codec 裝置",
                        choices=devices,
                        value=default_device,
                    )
                
                with gr.Row():
                    mula_dtype = gr.Dropdown(
                        label="MuLa 資料類型",
                        choices=["bfloat16", "float16", "float32"],
                        value="bfloat16",
                        info="bf16：效能較佳，記憶體用量較低",
                    )
                    
                    codec_dtype = gr.Dropdown(
                        label="Codec 資料類型",
                        choices=["bfloat16", "float16", "float32"],
                        value="float32",
                        info="fp32：音質較佳。使用 bf16 可能降低音質",
                    )
                
                lazy_load = gr.Checkbox(
                    label="Lazy Load（延遲載入）",
                    value=True,
                    info="啟用後模組將按需載入，可節省 GPU 記憶體",
                )

                quantize_4bit = gr.Checkbox(
                    label="4-bit 量化 (需 CUDA)",
                    value=False,
                    info="啟用 4-bit 量化可大幅減少 VRAM 使用量 (僅支援 CUDA)",
                )
                
                generate_btn = gr.Button("🎵 生成音樂", variant="primary", size="lg")
            
            # Right column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 輸出")
                
                audio_output = gr.Audio(
                    label="生成的音樂",
                    type="filepath",
                )
                
                gr.Markdown("### 📋 執行日誌")
                
                log_output = gr.Textbox(
                    label="日誌輸出",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    elem_classes=["log-box"],
                )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_music,
            inputs=[
                lyrics_input,
                tags_input,
                max_length,
                topk,
                temperature,
                cfg_scale,
                model_version,
                mula_device,
                codec_device,
                mula_dtype,
                codec_dtype,
                lazy_load,
                quantize_4bit,
            ],
            outputs=[audio_output, log_output],
        )
    
    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🚀 啟動 HeartLib Gradio 介面...")
    print(f"📁 輸出目錄: {os.path.abspath(OUTPUT_DIR)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
