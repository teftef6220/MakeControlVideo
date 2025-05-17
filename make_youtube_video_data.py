import sys
import os
sys.path.append(os.path.abspath("."))

from dotenv import load_dotenv
import json
import random
import numpy as np
import subprocess
import argparse
from yt_dlp import YoutubeDL

import torch
from tqdm import tqdm
from shutil import rmtree

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from dwpose import DwposeDetector, AnimalposeDetector

from Utils.create_data_utils import generate_caption, apply_dwpose
from Utils.path_utils import resolve_output_path , collect_video_paths


def download_and_clip_videos(args, json_path, output_dir, base_duration, delta):
    """
    YouTube またはローカルの動画をダウンロードして、指定された秒数で切り出す
    input
        args: コマンドライン引数
        json_path: 動画URLを含むJSONファイルのパス
        output_dir: 出力先ディレクトリ
        base_duration: 基準秒数 (n)
        delta: 許容誤差秒数 (±delta)
    output
        metadata_entries: 動画のメタデータを含むjsonファイルのパス
    """
    #### ================== 
    ##   0. Set up Directories
    #### ================== 
    # path チェック、この Ripo からの相対 path に変換
    resolve_output_path(output_dir)

    # output の親　path
    os.makedirs(output_dir, exist_ok=True)

    # 動画 clip path
    os.makedirs("temp", exist_ok=True)
    output_json_path = os.path.join(output_dir, args.metadata_json_name)
    output_clip_video_path = os.path.join(output_dir, "clip_video")
    os.makedirs(output_clip_video_path, exist_ok=True)

    # 必要ならば control path
    control_dir = os.path.join(output_dir, "control") if args.do_dwpose else None
    if control_dir:
        os.makedirs(control_dir, exist_ok=True)
    
    #### ================== 
    ##   1. Set up part
    #### ================== 
    # Load LLM
    if args.do_caption:
        bf16_supported = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        )
        model_path = os.path.join(os.environ['LLM_PATH'], args.llm_dir)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if bf16_supported else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

    # Load DWpose
    if args.do_dwpose:
        pose_model = DwposeDetector.from_pretrained(
            pretrained_model_or_path="yzd-v/DWPose",
            pretrained_det_model_or_path="yzd-v/DWPose",
            det_filename="yolox_l.onnx",
            pose_filename="dw-ll_ucoco_384.onnx",
            torchscript_device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    if args.use_wd14_tagger:
        pass
    #### ================== 
    ##   2. Download and clip videos part
    #### ================== 

    if args.use_youtube_data:
        print(f"Use Yoube mode. 📄 JSONファイルを読み込み中: {json_path}")
        with open(json_path, "r") as f:
            video_data = json.load(f)
        data_list = video_data["videos"]
    else:
        print(f"Use self data mode. 📂 自分の動画を読み込み中: {args.self_data_path}")
        self_data_list = collect_video_paths(args.self_data_path)
        data_list = self_data_list


    if args.data_count > 0:
        try:
            if os.path.exists(output_json_path):
                with open(output_json_path, "r", encoding="utf-8") as f:
                    metadata_entries = json.load(f)
                    video_index = len(metadata_entries) + 1
            else:
                print(f"❌ JSONファイルが存在しないか壊れています。新しいインデックスを1から開始します。")
                video_index = 1
        except (json.JSONDecodeError, IOError):
            # JSONファイルが壊れている・空・読み込みに失敗した場合
            print(f"❌ 読み込めませんでした。新しいインデックスを1から開始します。")
            video_index = 1
    else:
        video_index = 1


    metadata_entries = []
    for url in data_list:
        try:
            with YoutubeDL({
                'format': 'mp4',
                'outtmpl': 'temp/%(id)s.%(ext)s',
                'quiet': True,
            }) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = f"temp/{info['id']}.mp4"
                duration = info.get("duration", 0)

            result_video_path_list = []
            new_entries = []

            #### ================== 
            ##   2.1 Clip videos
            #### ================== 
            for _ in range(args.num_clips_per_videos):
                clip_duration = random.uniform(base_duration - delta, base_duration + delta)
                if duration <= clip_duration:
                    raise ValueError(f"URL {url} is too short for {clip_duration:.2f}s clipping.")

                start_time = random.uniform(0, duration - clip_duration)
                filename = f"{video_index:08d}.mp4"
                output_file_path = os.path.join(output_clip_video_path, filename)

                subprocess.run([
                    "ffmpeg", "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(clip_duration),
                    "-c:v", "libx264", "-c:a", "aac", "-y",
                    output_file_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                print(f"✅ Finish : {output_file_path}")
                result_video_path_list.append(output_file_path)

                entry = {
                    "file_path": os.path.join(os.environ['BASE_PATH'], output_file_path),
                    "type": "video",
                    "filename": filename,
                    "text": "",
                    "control_file_path": ""
                }
                new_entries.append(entry)
                video_index += 1

            # Generate captions
            if args.do_caption:
                captions = generate_caption(args, model, processor, result_video_path_list, os.environ['LLM_PATH'])
                for entry, caption in zip(new_entries, captions):
                    entry["text"] = caption

            # Apply DWpose
            if args.do_dwpose:
                control_paths = apply_dwpose(args, pose_model, result_video_path_list, control_dir)
                for entry, control_path in zip(new_entries, control_paths):
                    entry["control_file_path"] = os.path.join(os.environ['BASE_PATH'], control_path)

            # ✅ 成功したら保存
            metadata_entries.extend(new_entries)
            with open(output_json_path, "w") as f:
                json.dump(metadata_entries, f, indent=2)
            print(f"📄 保存: {output_json_path}")

            # temp の中身消す (temp は残す)
            if os.path.isfile(video_path):
                os.remove(video_path)
                print(f"{video_path} を削除しました。")
            else:
                print(f"{video_path} は存在しないか、ファイルではありません。")
        
        except Exception as e:
            print(f"❌ Skipped URL due to error: {url}")
            print(f"   Reason: {e}")
    
    # temp フォルダ削除
    print("temp フォルダを削除します。")
    rmtree("temp", ignore_errors=True)

    print(" ================= Finish !!!! ===================== ")
    print(f" make {video_index - 1} videos")
    print(f" metadata json file saved in : {output_json_path}")
    print(f" video clip saved in : {output_clip_video_path}")
    print(f" control video saved in : {control_dir}" if control_dir else "No control video")

if __name__ == "__main__":
    load_dotenv()
    print(os.environ['BASE_PATH'])
    
    parser = argparse.ArgumentParser(description="YouTube動画をダウンロードしてn±δ秒で切り出すツール")
    parser.add_argument('--use_youtube_data', type=bool, default=True, help="YouTube動画をダウンロードして切り出すか、既存の動画を切り出すか")
    parser.add_argument('--json',default='json_files/proto_test.json', help='動画URLを含むJSONファイルのパス')
    parser.add_argument('--self_data_path', default='path/to/your/self/data', help='自分の動画が保存されているディレクトリのパスyoutube を使うときは関係ない')
    parser.add_argument('--outdir', default='Result_data/proto_2', help='out put dir 相対path 推奨')
    parser.add_argument('--metadata_json_name', default='proto_2_metadeta.json', help='metadata json file')
    parser.add_argument('--n', type=float, default=1.0, help='基準秒数 (n)')
    parser.add_argument('--delta', type=float, default=0.0, help='許容誤差秒数 (±delta)')
    parser.add_argument('--num_clips_per_videos', type=int, default=2, help='各動画から切り出すクリップの数')
    parser.add_argument('--data_count', type=int, default=0, help='データに付けるindex番号のはじめを指定するか、1 以上を指定すると途中から判定される')

    ## LLM Caption
    parser.add_argument('--do_caption', type=bool, default=False,help="Do caotion generation")
    parser.add_argument('--llm_dir', default='Qwen2_5-VL-7B-Instruct', help='llm qwen 2.1 VL model dir')
    parser.add_argument('--prompt', type=str, default="Describe this video.", help='default prompt for caption generation')
    parser.add_argument('--dufault_caption', type=str, default="", help='default caption to add front of data')

    ## DWpose
    parser.add_argument('--do_dwpose', type=bool, default=True,help="Do dwpose estimation")

    ## WD14 Tagger
    parser.add_argument('--use_wd14_tagger', type=bool, default=False, help="Use wd14 tagger")
    
    args = parser.parse_args()
    download_and_clip_videos(args, args.json, args.outdir, args.n, args.delta)
