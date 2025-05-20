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
import cv2
import shutil 

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
from wd14_tagger.tagger import WD14VideoTagger


def download_and_clip_videos(args, json_path, output_dir, base_duration, delta):
    """
    YouTube またはローカルの動画をダウンロードして、指定された秒数で切り出す
    切り出すだけの長さがなかった場合に skipする
    DWpose , LLM , WD 14 タグはどれか一つでもエラーが出たらそのクリップは使わない
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
    # resolve_output_path(output_dir)

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
        wd14_model_dir = os.path.join(os.environ['BASE_PATH'],"wd14_tagger/wd14_models")
        wd14_model_name = "wd-vit-tagger-v3"
        tagger = WD14VideoTagger(model_dir=wd14_model_dir, model_name=wd14_model_name)
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
            if args.use_youtube_data:
                with YoutubeDL({
                    'format': 'mp4',
                    'outtmpl': 'temp/%(id)s.%(ext)s',
                    'quiet': True,
                }) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = f"temp/{info['id']}.mp4"
                    duration = info.get("duration", 0)
            else:
                # ローカル動画を temp にコピー
                original_path = url  # ローカルまたは既にDL済みの動画パス
                video_id = os.path.splitext(os.path.basename(original_path))[0]
                temp_path = os.path.join("temp", f"{video_id}.mp4")

                # temp にコピー（すでに存在していたらスキップするなども可能）
                shutil.copy2(original_path, temp_path)
                video_path = temp_path

                # 動画時間を計算
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {video_path}")
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()


            #### ================== 
            ##   2.1 Clip videos
            #### ================== 
            result_video_path_list = []
            new_entries = []
            for _ in range(args.num_clips_per_videos):
                try: ## クリップごとに pose 推定 cation 生成 → 両方成功しない限りそのクリップはスキップ
                    clip_duration = random.uniform(base_duration - delta, base_duration + delta)
                    if duration <= clip_duration:
                        # raise ValueError(f"URL {url} is too short for {clip_duration:.2f}s clipping.")
                        print(f"⚠️URL {url} is too short for {clip_duration:.2f}s clipping. スキップします。")
                        break

                    start_time = random.uniform(0, duration - clip_duration)
                    filename = f"{video_index:08d}.mp4"
                    output_file_path = os.path.join(output_clip_video_path, filename)

                    # クリップ生成
                    subprocess.run([
                        "ffmpeg", "-ss", str(start_time),
                        "-i", video_path,
                        "-t", str(clip_duration),
                        "-c:v", "libx264", "-c:a", "aac", "-y",
                        output_file_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    print(f"✅ Finish and Save : {output_file_path}")

                    entry = {
                        "file_path": os.path.join(os.environ['BASE_PATH'], output_file_path),
                        "type": "video",
                        "filename": filename,
                        "text": "",
                        "control_file_path": "",
                        "wd14_tag" : "",
                    }

                    wd14_tag, caption_text, control_path = "", "", ""

                    # WD14
                    if args.use_wd14_tagger:
                        tags = tagger.tag_videos([output_file_path], threshold=0.4, replace_underscore=False)
                        if not tags or not isinstance(tags[0], str):
                            raise ValueError("WD14 tagging failed.")
                        if args.use_wd14_tag_for_caption:
                            caption_text = tags[0]
                        else:
                            wd14_tag = tags[0]

                    # Qwen caption
                    if args.do_caption and not args.use_wd14_tag_for_caption:
                        captions = generate_caption(args, model, processor, [output_file_path], os.environ['LLM_PATH'])
                        if not captions or not isinstance(captions[0], str):
                            raise ValueError("Caption generation failed.")
                        caption_text = captions[0]

                    # DWpose
                    if args.do_dwpose:
                        control_paths,people_count_list,frame_count_list = apply_dwpose(args, pose_model, [output_file_path], control_dir)
                        detected_ratio = people_count_list[0] / frame_count_list[0]
                        control_path = os.path.join(os.environ['BASE_PATH'], control_paths[0])
                        if detected_ratio < args.detected_ratio_threshold:
                            raise ValueError("❌ DWpose detection ratio is too low. We Dont use this clip.")
                        if not control_paths or not os.path.exists(control_paths[0]):
                            raise ValueError("❌ DWpose failed.")
                        

                    # ✅ 必要な情報がそろったので entry を構築
                    entry = {
                        "file_path": os.path.join(os.environ['BASE_PATH'], output_file_path),
                        "type": "video",
                        "filename": filename,
                    }

                    if caption_text:
                        entry["text"] = caption_text
                    if wd14_tag:
                        entry["wd14_tag"] = wd14_tag
                    if control_path:
                        entry["control_file_path"] = control_path

                    new_entries.append(entry)
                    video_index += 1

                except Exception as e:
                    print("⚠️ 1クリップでエラーが発生しました。このクリップはスキップします。")
                    print(f"   Reason: {e}")

                    # 失敗したクリップ動画を削除
                    try:
                        if os.path.exists(output_file_path):
                            os.remove(output_file_path)
                            print(f"🗑️ クリップ削除: {output_file_path}")
                        if os.path.exists(control_path):
                            os.remove(control_path)
                            print(f"🗑️ Control file削除: {control_path}")
                    except Exception as remove_error:
                        print(f"❌ 削除失敗: {output_file_path}, 理由: {remove_error}")

            # ✅ 成功したらjson保存
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
    parser.add_argument('--use_youtube_data', type=bool, default=False, help="YouTube動画をダウンロードして切り出すか、既存の動画を切り出すか")
    parser.add_argument('--json',default='json_files/proto_test.json', help='動画URLを含むJSONファイルのパス')
    parser.add_argument('--self_data_path', default='/path/to/image/dir/', help='自分の動画が保存されているディレクトリのパス,youtube を使うときは関係ない')
    parser.add_argument('--outdir', default='/path/to/out/dir/', help='出力 output dir 絶対 path 推奨')
    parser.add_argument('--metadata_json_name', default='test_metadeta.json', help='metadata json file')
    parser.add_argument('--n', type=float, default=3.0, help='基準秒数 (n)')
    parser.add_argument('--delta', type=float, default=0.0, help='許容誤差秒数 (±delta)')
    parser.add_argument('--num_clips_per_videos', type=int, default=1, help='各動画から切り出すクリップの数')
    parser.add_argument('--data_count', type=int, default=0, help='データに付けるindex番号のはじめを指定するか、1 以上を指定すると途中から判定される')

    ## LLM Caption
    parser.add_argument('--do_caption', type=bool, default=False ,help="Do caotion generation")
    parser.add_argument('--llm_dir', default='Qwen2_5-VL-7B-Instruct', help='llm qwen 2.1 VL model dir')
    parser.add_argument('--prompt', type=str, default="Describe this video.", help='default prompt for caption generation')
    parser.add_argument('--default_caption', type=str, default="", help='default caption to add front of data')

    ## DWpose
    parser.add_argument('--do_dwpose', type=bool, default=True,help="Do dwpose estimation")
    parser.add_argument('--add_occlusion', type=bool, default=False, help="Add occlusion to dwpose ,shift and scale the pose") ## TODO 実装待ち
    parser.add_argument('--detected_ratio_threshold', type=float, default=0.3, help="DWpose detection ratio threshold,検出された人数が少なすぎる場合はその動画は破棄") 

    ## WD14 Tagger
    parser.add_argument('--use_wd14_tagger', type=bool, default=True, help="Use wd14 tagger")
    parser.add_argument('--use_wd14_tag_for_caption', type=bool, default=True, help="Use wd14 tagger for caption generation")
    
    args = parser.parse_args()
    download_and_clip_videos(args, args.json, args.outdir, args.n, args.delta)
