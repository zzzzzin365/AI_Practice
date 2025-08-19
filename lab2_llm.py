# 根据你的系统，可以提前在本地安装好 ffmpeg
%pip install genai
%pip install pydub
%pip install requests

#初始化客户端
import os
import json
from google import genai
from google.colab import userdata

GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')
SILICONFLOW_API_KEY=userdata.get('SILICONFLOW_API_KEY')

client = genai.Client(
  api_key=GEMINI_API_KEY
)

TTS_API_CONFIG = {
    "url": "https://api.siliconflow.cn/v1/audio/speech",
    "token": SILICONFLOW_API_KEY,  # 请替换为你的实际token
    "default_params": {
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,  # 改为False获取完整响应
        "speed": 1.1,
        "gain": 0,
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "voice": "FunAudioLLM/CosyVoice2-0.5B:alex"
    }
}

#请求 Gemini API 传入视频返回第一轮切片数据
import json

file_url = "./test.mp4"

prompt = """
## 总体要求
- 帮我从这个视频当中提取出来 5-15 个片段，
- 这些片段会根据下面的描述来重新表达合适的视频时间序列组合，方便后续进行视频的重新拼接。

## 片段要求
- 每个片段的开始时间和结束时间，
- 视频拼接要有网感，不要出现重复的片段，
- 这些片段能组合出来一个适合推广户外空调的口播视频，总时间长度在 15-30s 最佳。

## 输出格式
请返回 json 格式数据，数据格式里除了包含开始和结束时间，
开始时间用 start_time 表示，结束时间用 end_time 表示，
数据结构如下：
{
  "total_duration_estimate": "28s",
  "clip_count": 8,
  "clips": [
    {
      "clip_id": 1,
      "start_time": "00:00",
      "end_time": "00:04",
      "visual_description": "【热到融化】特写夏日户外烈日炎炎，人们汗流浃背，表情痛苦挣扎，甚至出现夸张的“蒸发”效果。网感：‘这天气，出门就是铁板烧！’"
    },
    ...
  ]
}
"""


response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[
      file_url,
      prompt
    ]
)

original_text = response.text

# 提取 json 数据，```json 开头，``` 结尾
json_data = original_text.split("```json")[1].split("```")[0]
video_chunk_data = json.loads(json_data)

# 美化输出 JSON 数据
print(json.dumps(video_chunk_data, indent=2, ensure_ascii=False))

#功能代码（第一阶段）
import subprocess

def time_to_seconds(time_str):
    """将时间字符串转换为秒数 (支持 MM:SS 和 MM:SS.mmm 格式)"""
    if ':' not in time_str:
        return float(time_str)

    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS 格式
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS 格式
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        return float(time_str)

def extract_segment_with_audio(video_path, start_time, end_time, output_path, trim_seconds=0.1):
    """
    使用ffmpeg提取视频片段（包含音频），并可选择裁切首尾
    trim_seconds: 从开头和结尾各裁切的秒数，有助于消除拼接时的停顿感
    """
    # 计算实际的裁切时间
    actual_start = start_time + trim_seconds
    actual_end = end_time - trim_seconds

    # 确保裁切后还有足够的内容
    if actual_end <= actual_start:
        print(f"警告: 片段太短，无法裁切 {trim_seconds}s，使用原始时间")
        actual_start = start_time
        actual_end = end_time

    duration = actual_end - actual_start

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(actual_start),
        '-t', str(duration),
        '-c:v', 'libx264',      # 重新编码视频确保兼容性
        '-c:a', 'aac',          # 重新编码音频确保兼容性
        '-r', '25',             # 强制帧率为25fps
        '-preset', 'fast',      # 编码速度
        '-crf', '23',           # 质量设置
        '-avoid_negative_ts', 'make_zero',
        '-fflags', '+genpts',   # 重新生成时间戳
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"提取片段成功: {output_path} (裁切了 {trim_seconds}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg错误: {e.stderr}")
        return False
    except FileNotFoundError:
        print("错误: 找不到ffmpeg，请先安装ffmpeg")
        return False

def convert_to_vertical_simple(input_path, output_path, target_size=(1080, 1920)):
    """使用更简单的方法转换为竖屏"""
    target_width, target_height = target_size

    # 最简单的方法：直接resize并添加黑边
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2",
        '-c:a', 'copy',
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"转换竖屏成功 (简单模式): {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg简单模式也失败: {e.stderr}")
        return False

def convert_to_vertical_ffmpeg(input_path, output_path, target_size=(1080, 1920), bg_color=(0, 0, 0)):
    """使用ffmpeg将横屏视频转换为竖屏"""
    target_width, target_height = target_size

    # 更简单的滤镜方案：先缩放，再添加黑边
    # 1. 计算缩放比例，保持宽高比
    # 2. 使用pad添加黑边到目标尺寸
    filter_complex = f"scale='min({target_width},iw)':'min({target_height},ih)':force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"

    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', filter_complex,
        '-c:a', 'copy',  # 音频不重新编码
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"转换竖屏成功: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg错误: {e.stderr}")
        # 如果还是失败，尝试更简单的方法
        return convert_to_vertical_simple(input_path, output_path, target_size)

def concatenate_videos_with_transitions(video_paths, output_path, use_crossfade=False):
    """
    使用ffmpeg拼接视频，添加转场效果减少停顿感
    use_crossfade: 是否使用交叉淡化转场
    """
    # 使用简单拼接但确保流畅性
    list_file = "video_list.txt"
    with open(list_file, 'w', encoding='utf-8') as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c:v', 'libx264',      # 重新编码确保一致性
        '-c:a', 'aac',
        '-r', '25',             # 统一帧率
        '-preset', 'fast',
        '-crf', '23',
        '-fflags', '+genpts',   # 重新生成时间戳
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"视频拼接成功: {output_path}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg错误: {e.stderr}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


def process_video_only(video_path, video_chunk_data, output_path, target_size=(1080, 1920), trim_seconds=0.1):
    """
    只处理视频部分，不处理音频
    """
    clips = video_chunk_data['clips']

    # 创建临时目录
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)

    processed_video_paths = []

    print(f"🎬 开始处理 {len(clips)} 个视频片段...")

    try:
        for i, clip in enumerate(clips):
            clip_id = clip['clip_id']
            start_time = time_to_seconds(clip['start_time'])
            end_time = time_to_seconds(clip['end_time'])

            print(f"\n--- 处理片段 {clip_id}/{len(clips)} ---")
            print(f"时间: {start_time}s - {end_time}s")
            print(f"画面: {clip['visual_description']}")

            # 1. 提取视频片段
            segment_path = os.path.join(temp_dir, f"segment_{clip_id}.mp4")
            if not extract_segment_with_audio(video_path, start_time, end_time, segment_path, trim_seconds):
                print(f"跳过片段 {clip_id}: 提取失败")
                continue

            # 2. 转换为竖屏
            vertical_path = os.path.join(temp_dir, f"vertical_{clip_id}.mp4")
            if not convert_to_vertical_ffmpeg(segment_path, vertical_path, target_size):
                print(f"跳过片段 {clip_id}: 竖屏转换失败")
                continue

            processed_video_paths.append(vertical_path)

        if not processed_video_paths:
            raise ValueError("没有成功处理的视频片段")

        # 3. 拼接所有处理好的片段
        print(f"\n🎬 拼接 {len(processed_video_paths)} 个片段...")
        if not concatenate_videos_with_transitions(processed_video_paths, output_path, use_crossfade=False):
            raise ValueError("视频拼接失败")

        print(f"\n✅ 视频处理完成!")
        print(f"输出文件: {output_path}")

        return True

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return False

    finally:
        # 清理临时文件
        print("\n🧹 清理临时文件...")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"清理文件失败 {file_path}: {e}")

            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"清理目录失败: {e}")

#功能代码（第二阶段）
import requests
from pydub import AudioSegment

def generate_voiceover_script_with_gemini(video_path):
    """
    使用Gemini生成配音文案
    基于新合成的视频内容配合 Prompt 生成要求的文案
    """

    prompt = """
    帮我基于当前的视频来生成一份适合的口播带货文案，核心要突出户外空调的卖点，
    要求时间匹配要清晰，文案简短有力。
    每秒最长能读 4 个字左右，按照这个来参考文案长度。
    返回 json 格式，包含每个时间段的文案和对应的时间戳。
    格式如：{'segments': [{'start_time': '00:00', 'end_time': '00:02', 'script': '文案内容'}]}
    """
    print("🤖 正在使用Gemini生成配音文案...")

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            video_path,
            prompt
        ]
    )

    print("✅ Gemini文案生成完成")
    return response.text

def extract_gemini_script(gemini_response):
    """从Gemini响应中提取JSON格式的文案"""
    try:
        # 提取 ```json 开头，``` 结尾的内容
        if "```json" in gemini_response and "```" in gemini_response:
            json_data = gemini_response.split("```json")[1].split("```")[0]
            script_data = json.loads(json_data)
            print("✅ 文案数据解析成功")
            return script_data
        else:
            print("❌ 未找到JSON格式的文案数据")
            return None
    except Exception as e:
        print(f"❌ 文案数据解析失败: {e}")
        return None

def generate_tts_audio(text, output_path, clip_id=None):
    """调用TTS API生成语音"""
    try:
        headers = {
            "Authorization": f"Bearer {TTS_API_CONFIG['token']}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            **TTS_API_CONFIG["default_params"]
        }

        print(f"正在生成语音 clip_{clip_id}: {text[:50]}...")

        response = requests.post(
            TTS_API_CONFIG["url"],
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            # 保存音频文件
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ 语音生成成功: {output_path}")
            return True
        else:
            print(f"❌ TTS API错误 clip_{clip_id}: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ 语音生成异常 clip_{clip_id}: {e}")
        return False

def replace_audio_in_video(video_path, audio_path, output_path):
    """替换视频中的音频"""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # 视频不重新编码
        '-c:a', 'aac',   # 音频编码为AAC
        '-map', '0:v:0', # 使用第一个输入的视频流
        '-map', '1:a:0', # 使用第二个输入的音频流
        '-shortest',     # 以较短的流为准
        '-y',
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"音频替换成功: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg错误: {e.stderr}")
        return False


def generate_audio_from_script_and_merge(video_path, script_data, output_path, audio_dir="./script_audio"):
    """根据文案生成音频并合成到视频"""
    os.makedirs(audio_dir, exist_ok=True)

    if not script_data or 'segments' not in script_data:
        print("❌ 无效的文案数据")
        return False

    segments = script_data['segments']
    audio_files = []

    print(f"🎵 开始生成 {len(segments)} 个文案音频...")

    # 1. 生成所有音频片段
    for i, segment in enumerate(segments):
        script_text = segment['script']
        audio_filename = f"script_{i+1}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)

        if generate_tts_audio(script_text, audio_path, f"script_{i+1}"):
            audio_files.append(audio_path)
        else:
            # 如果生成失败，创建静音
            start_time = time_to_seconds(segment['start_time'])
            end_time = time_to_seconds(segment['end_time'])
            duration = end_time - start_time

            silence = AudioSegment.silent(duration=int(duration * 1000))
            silence_path = os.path.join(audio_dir, f"silence_{i+1}.wav")
            silence.export(silence_path, format="wav")
            audio_files.append(silence_path)

    if not audio_files:
        print("❌ 没有生成任何音频文件")
        return False

    # 2. 拼接所有音频
    print("🎵 拼接音频文件...")
    combined_audio = AudioSegment.empty()
    for audio_file in audio_files:
        try:
            if audio_file.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(audio_file)
            else:
                audio_segment = AudioSegment.from_wav(audio_file)
            combined_audio += audio_segment
        except Exception as e:
            print(f"⚠️ 音频文件加载失败 {audio_file}: {e}")

    # 3. 保存合并的音频
    combined_audio_path = os.path.join(audio_dir, "combined_script_audio.wav")
    combined_audio.export(combined_audio_path, format="wav")
    print(f"✅ 音频拼接完成: {combined_audio_path}")

    # 4. 将音频合成到视频
    print("🎬 将音频合成到视频...")
    if replace_audio_in_video(video_path, combined_audio_path, output_path):
        print(f"✅ 最终视频生成完成: {output_path}")
        return True
    else:
        print("❌ 音频合成失败")
        return False

#流程代码
from datetime import datetime

# 文件路径
input_video_path = "./test.mp4"
temp_video_path = f"./temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
output_video_path = f"./final_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

# 竖屏设置
target_size = (1080, 1920)  # 9:16 比例


# 第一阶段：只处理视频，不处理音频
video_success = process_video_only(
    video_path=input_video_path,
    video_chunk_data=video_chunk_data,
    output_path=temp_video_path,
    target_size=target_size,
    trim_seconds=0.2
)

# 第二阶段：使用Gemini生成文案
gemini_response = generate_voiceover_script_with_gemini(temp_video_path)

# 解析文案
script_data = extract_gemini_script(gemini_response)

if script_data:
    print("\n" + "="*60)
    print("🎵 第三阶段：生成配音并合成...")
    print("="*60)

    # 第三阶段：根据文案生成音频并合成
    final_success = generate_audio_from_script_and_merge(
        video_path=temp_video_path,
        script_data=script_data,
        output_path=output_video_path,
        audio_dir="./script_audio"
    )

    if final_success:
        # 清理临时视频文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"✅ 清理临时文件: {temp_video_path}")

        # 保存文案信息
        script_file = f"script_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(script_file, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)

        print(f"\n📄 文案信息已保存: {script_file}")

        # 保存原始项目信息
        info_file = f"video_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(video_chunk_data, f, ensure_ascii=False, indent=2)

        print(f"📄 项目信息已保存: {info_file}")

    else:
        print("❌ 配音合成失败，但视频文件已生成:", temp_video_path)
        output_video_path = temp_video_path  # 使用临时视频作为最终输出
else:
    print("❌ 文案解析失败，但视频文件已生成:", temp_video_path)
    output_video_path = temp_video_path  # 使用临时视频作为最终输出

print("\n" + "="*60)
print("🎉 处理完成！")
print("="*60)