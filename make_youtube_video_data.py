import sys
import os
sys.path.append(os.path.abspath("."))

from dotenv import load_dotenv
import json
import random
import subprocess
import argparse
from yt_dlp import YoutubeDL

import torch
import uuid
from tqdm import tqdm
import shutil
from shutil import rmtree
import subprocess

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

from dwpose import DwposeDetector, AnimalposeDetector
import cv2
import numpy as np

def generate_caption(args, result_video_path_list, llm_base_dir):
    """
    Generate captions for videos using a pre-trained qwen2.5 VL model.
    input
        args: command line arguments
        result_video_path_list: list of video file paths
        llm_base_dir: model name , base directory where the model is stored
    return
        result_messages_list: list of generated captions for each video

    if there are errors in the video, it will be skipped and continue to the next video.
    not saving caption in result_video_path_list
    """
    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )

    model_path = os.path.join(llm_base_dir, args.llm_dir)

    # モデルとプロセッサを1回だけロード
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if bf16_supported else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    result_messages_list = []

    ## inference loop
    print(f"================= Start Gen Caption Use {args.llm_dir} =====================")
    for path in tqdm(result_video_path_list, desc="Generating captions"):

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{os.path.abspath(path)}",
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ]


            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                # fps=1.0,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to("cuda")

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result_messages_list.append(output_text)
        except Exception as e:
            print(f"❌ エラー: {path} -> {e}")
            continue

    return result_messages_list

    
def apply_dwpose(args,video_path_list, control_dir):
    """
    Apply DWpose to the list of videos and save the output videos in the control_dir.
    
    input
        args: command line arguments
        video_path_list: list of video file paths
        control_dir: directory to save the output videos (relative to the current working directory)
    
    return
        result_control_paths: list of paths to the output videos

    if there are errors in the video, it will be skipped and continue to the next video.
    not saving control video in result_control_paths
    """
    os.makedirs(control_dir, exist_ok=True)

    # モデルの読み込み（1回だけ）
    pose_model = DwposeDetector.from_pretrained(
    pretrained_model_or_path="yzd-v/DWPose",  # モデルの保存先 or huggingface repo
    pretrained_det_model_or_path="yzd-v/DWPose",  # bboxモデル保存先（同じでもOK）
    det_filename="yolox_l.onnx",
    pose_filename="dw-ll_ucoco_384.onnx",
    torchscript_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    result_control_paths = []


    print (f"================= Start Gen DWpose =====================")
    
    for video_path in tqdm(video_path_list, desc="Estimating pose"):

        try:
            video_name = os.path.basename(video_path)
            output_video_path = os.path.join(control_dir, video_name)

            tmp_frame_dir = f"temp_dwpose_frames_{video_name}"
            os.makedirs(tmp_frame_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 24.0  # fallback
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            frame_paths = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_image = pose_model(rgb_frame,include_body=True, include_hand=True, include_face=True)
                pose_image = np.array(pose_image)

                if (pose_image.shape[1], pose_image.shape[0]) != frame_size:
                    pose_image = cv2.resize(pose_image, frame_size, interpolation=cv2.INTER_LINEAR)

                frame_out_path = os.path.join(tmp_frame_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_out_path, cv2.cvtColor(pose_image, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_out_path)
                frame_idx += 1

            cap.release()

            # 書き出しに ffmpeg を使用
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # 上書き
                "-framerate", str(fps),
                "-i", os.path.join(tmp_frame_dir, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"❌ ffmpeg failed--------------------------: {e}")

            # 一時フォルダ削除
            rmtree(tmp_frame_dir, ignore_errors=True)

            # 動画のパスを保存
            result_control_paths.append(output_video_path)

        except Exception as e:
            print(f"❌ エラー: {video_path} -> {e}")

            # if erro remove from result_control_paths
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
                result_control_paths.remove(output_video_path)
                rmtree(tmp_frame_dir, ignore_errors=True)
            continue

    return result_control_paths

def download_and_clip_videos(args, json_path, output_dir, base_duration, delta):
    os.makedirs(output_dir, exist_ok=True)
    control_dir = os.path.join(output_dir, "control") if args.do_dwpose else None
    if control_dir:
        os.makedirs(control_dir, exist_ok=True)

    os.makedirs("temp", exist_ok=True)

    metadata_entries = []

    # 1. 全動画を先にダウンロード & 切り出し
    with open(json_path, "r") as f:
        video_data = json.load(f)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': 'temp/%(id)s.%(ext)s',
        'quiet': True,
    }

    video_index = 1

    with YoutubeDL(ydl_opts) as ydl:
        for url in video_data["videos"]:
            try:
                info = ydl.extract_info(url, download=True)
                video_path = f"temp/{info['id']}.mp4"
                duration = info.get("duration", 0)

                for _ in range(args.num_clips_per_videos):
                    clip_duration = random.uniform(base_duration - delta, base_duration + delta)
                    if duration <= clip_duration:
                        print(f"Skip: {url} is shorter than {clip_duration:.2f}s.")
                        break

                    start_time = random.uniform(0, duration - clip_duration)
                    filename = f"{video_index:08d}.mp4"
                    output_file_path = os.path.join(output_dir, filename)

                    subprocess.run([
                        "ffmpeg", "-ss", str(start_time),
                        "-i", video_path,
                        "-t", str(clip_duration),
                        "-c:v", "libx264", "-c:a", "aac", "-y",
                        output_file_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    print(f"✅ Finish : {output_file_path}")

                    entry = {
                        "file_path": os.path.join(os.environ['BATH_PATH'],output_file_path),
                        "type": "video",
                        "filename": filename,
                        # caption や control_file_path は後で追加
                    }
                    metadata_entries.append(entry)
                    video_index += 1

            except Exception as e:
                print(f"❌ エラー: {url} -> {e}")

    # 2. json 出力先を決定 
    output_json_path = os.path.join(output_dir , "metadata.json")
    with open(output_json_path, "w") as f:
        json.dump(metadata_entries, f, indent=2)
    print(f"\n📄 ファイル path だけの metadata.json を出力: {output_json_path}")

    # 3. 動画のパスを取得
    result_video_path_list = [
        os.path.join(output_dir, entry["filename"])
        for entry in metadata_entries
    ]

    # 4. キャプションを一括生成して反映 (optional)
    if args.do_caption:
        captions = generate_caption(args, result_video_path_list, os.environ['LLM_PATH'])
        for entry, caption in zip(metadata_entries, captions):
            entry["text"] = caption[0]

        with open(output_json_path, "w") as f:
            json.dump(metadata_entries, f, indent=2)
        print(f"\n📄 ファイル path とcaption の metadata.json を出力: {output_json_path}")

    # 5. DWpose を一括生成して反映 (optional)
    if args.do_dwpose:
        control_paths = apply_dwpose(args, result_video_path_list, control_dir)
        for entry, control_path in zip(metadata_entries, control_paths):
            # "control/00000001.mp4" のように保存したい場合、相対パスに変換
            entry["control_file_path"] = os.path.join(os.environ['BATH_PATH'],control_path)

        with open(output_json_path, "w") as f:
            json.dump(metadata_entries, f, indent=2)
        print(f"\n📄 ファイル path とcaption と controlvideo path だけの metadata.json を出力: {output_json_path}")

    # 5. temp 削除などの後処理
    shutil.rmtree("temp", ignore_errors=True)

    print(f"\n\n\n================= Finish =====================")
    print(f"📁 動画の保存先: {output_dir}")



if __name__ == "__main__":


    load_dotenv()
    print(os.environ['BATH_PATH'])
    
    parser = argparse.ArgumentParser(description="YouTube動画をダウンロードしてn±δ秒で切り出すツール")
    parser.add_argument('--json',default='json_files/proto.json', help='動画URLを含むJSONファイルのパス')
    parser.add_argument('--outdir', default='Result_data/proto_1', help='out put dir')
    parser.add_argument('--n', type=float, default=3.0, help='基準秒数 (n)')
    parser.add_argument('--delta', type=float, default=1.0, help='許容誤差秒数 (±delta)')
    parser.add_argument('--num_clips_per_videos', type=int, default=2, help='各動画から切り出すクリップの数')

    parser.add_argument('--do_caption', type=bool, default=True,help="Do caotion generation")
    parser.add_argument('--llm_dir', default='Qwen2_5-VL-7B-Instruct', help='llm qwen 2.1 VL model dir')

    parser.add_argument('--do_dwpose', type=bool, default=True,help="Do dwpose estimation")

    args = parser.parse_args()
    download_and_clip_videos(args, args.json, args.outdir, args.n, args.delta)
