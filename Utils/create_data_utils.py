
import subprocess
from tqdm import tqdm
from shutil import rmtree
import subprocess
import numpy as np
import cv2
import os

from qwen_vl_utils import process_vision_info




def apply_dwpose(args,pose_model,video_path_list, control_dir):
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

    result_control_paths = []
    result_total_frames = []
    result_people_count = []

    print (f"================= Start Gen DWpose =====================")
    
    for video_path in tqdm(video_path_list, desc="Estimating pose"):

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

        people_count = 0
        total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_image, keypoint_dict = pose_model(rgb_frame,include_body=True, include_hand=True, include_face=True, image_and_json=True)
            pose_image = np.array(pose_image)

            if (pose_image.shape[1], pose_image.shape[0]) != frame_size:
                pose_image = cv2.resize(pose_image, frame_size, interpolation=cv2.INTER_LINEAR)

            frame_out_path = os.path.join(tmp_frame_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_out_path, cv2.cvtColor(pose_image, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_out_path)
            frame_idx += 1

            if len(keypoint_dict.get("people", [])) > 0:
                people_count += 1

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

        result_people_count.append(people_count)
        result_total_frames.append(total_frames)

    return result_control_paths , result_people_count, result_total_frames



def generate_caption(args,model, processor,result_video_path_list, llm_base_dir):
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

    result_messages_list = []

    ## inference loop
    print(f"================= Start Gen Caption Use {args.llm_dir} =====================")
    for path in tqdm(result_video_path_list, desc="Generating captions"):

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
                        {"type": "text", "text": args.prompt},
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

            if args.default_caption != "":
                result_text = args.default_caption + output_text[0]
            else:
                result_text = output_text[0]

            result_messages_list.append(result_text)


    return result_messages_list