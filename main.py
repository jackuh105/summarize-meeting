import io
import os
import httpx
import asyncio
import argparse
import warnings
from dotenv import load_dotenv
import torch
import torchaudio
import pyannote.audio.core.task
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from concurrent.futures import ThreadPoolExecutor, as_completed

# ignore torchaudio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# settings
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)
PADDING = 0.3
MAX_WORKERS = 3
STT_TIMEOUT = 60.0

# add pyannote.audio to torch's safe global list
torch.serialization.add_safe_globals([
    pyannote.audio.core.task.Specifications,
    pyannote.audio.core.task.Problem,
    pyannote.audio.core.task.Resolution,
])
# setup pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN")
)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
pipeline.to(device)

def get_speaker_tracks(output):
    """
    Get speaker tracks from output
    """
    speaker_tracks = {}
    for turn, speaker in output.speaker_diarization:
        if speaker not in speaker_tracks:
            speaker_tracks[speaker] = []
        speaker_tracks[speaker].append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return speaker_tracks

def merge_segments(speaker_tracks, max_gap=2.0):
    """
    Merge segments that are close to each other
    """
    merged_results = []
    for speaker, segments in speaker_tracks.items():
        # ensure segments are sorted by start time
        segments.sort(key=lambda x: x["start"])
        if not segments:
            continue

        current_segment = segments[0]
        for next_segment in segments[1:]:
            gap = next_segment["start"] - current_segment["end"]
            if gap <= max_gap:
                current_segment["end"] = max(current_segment["end"], next_segment["end"])
            else:
                merged_results.append(current_segment)
                current_segment = next_segment
        merged_results.append(current_segment)
    merged_results.sort(key=lambda x: x["start"])
    return merged_results

def prepare_audio_buffer(waveform, start_time, end_time, sample_rate):
    try:
        total_frames = waveform.shape[1]
        s_time = max(0, start_time - PADDING)
        e_time = end_time + PADDING
        start_frame = int(s_time * sample_rate)
        end_frame = int(e_time * sample_rate)
        if end_frame > total_frames: end_frame = total_frames
        segment_waveform = waveform[:, start_frame:end_frame]
        buffer = io.BytesIO()
        torchaudio.save(buffer, segment_waveform, sample_rate, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error in prepare_audio_buffer: {e}")
        return None

async def transcribe_segment(sem, client, segment, waveform, sample_rate):
    async with sem:
        try:
            buffer = await asyncio.to_thread(
                prepare_audio_buffer,
                waveform,
                segment["start"],
                segment["end"],
                sample_rate
            )
            if buffer is None:
                return {**segment, "text": "[Audio Error]"}

            files = { 'file': (f"{segment['speaker']}.wav", buffer, 'audio/wav') }
            data = {
                'model_name': 'sensevoice',
                'response_format': 'json',
                'temperature': 0.0
            }
            response = await client.post("/audio/transcriptions", files=files, data=data) # openai api compatible format
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                return {**segment, "text": text}
            else:
                print(f"Error transcribing segment: {response.status_code}")
                return {**segment, "text": "[Transcribe Error]"}
        except Exception as e:
            print(f"Error processing segment: {segment['start']:.1f}s: {e}")
            return {**segment, "text": "[Exception]"}

def generate_srt(segments):
    srt_output = []
    for i, seg in enumerate(segments, 1):
        def format_time(seconds):
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            mins, secs = divmod(seconds, 60)
            hours, mins = divmod(mins, 60)
            return f"{hours:02}:{mins:02}:{secs:02},{millis:03}"

        start_str = format_time(seg['start'])
        end_str = format_time(seg['end'])
        
        srt_block = f"{i}\n{start_str} --> {end_str}\n[{seg['speaker']}] {seg['text']}\n"
        srt_output.append(srt_block)
    
    return "\n".join(srt_output)

def export_audio_segments(audio_path, segments, output_dir="segments", padding=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    waveform, sample_rate = torchaudio.load(audio_path)
    total_frames = waveform.shape[1]
    for seg in segments:
        start_time = max(0, seg["start"] - padding)
        end_time = seg["end"] + padding

        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        if end_frame > total_frames: end_frame = total_frames

        segment_waveform = waveform[:, start_frame:end_frame]
        original_duration = seg["end"] - seg["start"]
        filename = f"{seg['speaker']}_{seg['start']:.1f}_{seg['end']:.1f}_{original_duration:.1f}s.wav"
        save_path = os.path.join(output_dir, filename)
        torchaudio.save(save_path, segment_waveform, sample_rate)

async def main(input_path):
    print("1. Running Diarization...")
    with ProgressHook() as hook:
        output = pipeline(input_path, hook=hook)

    print("2. Merging Segments...")
    speaker_tracks = get_speaker_tracks(output)
    cleaned_tracks = merge_segments(speaker_tracks, max_gap=2.0)

    print(f"3. Loading Audio into Memory ({input_path})...")
    waveform, sample_rate = torchaudio.load(input_path)

    print(f"4. Transcribing {len(cleaned_tracks)} segments (Parallel Workers={MAX_WORKERS})...")
    final_segments = []
    sem = asyncio.Semaphore(MAX_WORKERS)
    async with httpx.AsyncClient(base_url=os.getenv("STT_BASE_API"), timeout=STT_TIMEOUT) as client:
        tasks = [
            transcribe_segment(sem, client, seg, waveform, sample_rate)
            for seg in cleaned_tracks
        ]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            seg = await task
            final_segments.append(seg)
            if i % 5 == 0 or (i + 1) == len(cleaned_tracks):
                print(f"Processed {i+1}/{len(cleaned_tracks)} segments...")
    final_segments.sort(key=lambda x: x["start"])

    print("5. Generating SRT...")
    srt_content = generate_srt(final_segments)
    output_srt_path = input_path.replace(".wav", ".srt")
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"Done! SRT saved to: {output_srt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input audio file path")
    args = parser.parse_args()

    asyncio.run(main(args.input))
