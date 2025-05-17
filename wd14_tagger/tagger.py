import os
import csv
import numpy as np
from PIL import Image
import onnxruntime as ort
import requests
from onnxruntime import InferenceSession
import cv2

class WD14VideoTagger:
    """
    A class to tag images and videos using the WD14 model.
    """
    def __init__(
        self,
        model_dir: str,
        model_name: str = "wd-v1-4-moat-tagger-v2", ## wd-v1-4-moat-tagger-v2 or wd-vit-tagger-v3 is recommended
        providers: list = None,
        hf_endpoint: str = "https://huggingface.co/SmilingWolf"
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.hf_endpoint = hf_endpoint
        self.session, self.tags, self.general_index, self.character_index = self._load_model()

    def _download_file(self, url, output_path):
        print(f"Downloading {url} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {output_path}")

    def _load_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.model_name}.onnx")
        label_path = os.path.join(self.model_dir, f"{self.model_name}.csv")
        base_url = f"{self.hf_endpoint}/{self.model_name}/resolve/main"

        if not os.path.exists(model_path):
            self._download_file(f"{base_url}/model.onnx", model_path)
        if not os.path.exists(label_path):
            self._download_file(f"{base_url}/selected_tags.csv", label_path)

        session = ort.InferenceSession(model_path, providers=self.providers)

        tags = []
        general_index = None
        character_index = None
        with open(label_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])

        return session, tags, general_index, character_index

    def _extract_frame(self, video_path: str, frame_index: int = 0) -> Image.Image:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = min(frame_index, total - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"フレームの読み込みに失敗しました: {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _preprocess_image(self, image: Image.Image, target_size: int = 448) -> np.ndarray:
        ratio = float(target_size) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        square.paste(image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
        array = np.array(square)[:, :, ::-1].astype(np.float32)  # RGB → BGR
        return np.expand_dims(array, 0)

    def tag_image(
        self,
        image: Image.Image,
        threshold=0.35,
        character_threshold=0.85,
        exclude_tags=[],
        replace_underscore=False,
        trailing_comma=False
    ) -> str:
        input_name = self.session.get_inputs()[0].name
        label_name = self.session.get_outputs()[0].name
        input_tensor = self._preprocess_image(image)

        probs = self.session.run([label_name], {input_name: input_tensor})[0][0]

        tags_list = []
        for idx, prob in enumerate(probs):
            tag = self.tags[idx].replace("_", " ") if replace_underscore else self.tags[idx]
            if idx >= self.character_index:
                if prob > character_threshold:
                    tags_list.append(tag)
            elif idx >= self.general_index:
                if prob > threshold:
                    tags_list.append(tag)

        tags_list = [t for t in tags_list if t not in exclude_tags]

        if trailing_comma:
            return ", ".join(t + "," for t in tags_list)
        else:
            return ", ".join(tags_list)

    def tag_videos(
        self,
        result_video_path_list,
        threshold=0.35,
        character_threshold=0.85,
        exclude_tags=[],
        replace_underscore=False,
        trailing_comma=False,
        frame_index=0
    ) -> list:
        results = []

        for video_path in result_video_path_list:
            try:
                image = self._extract_frame(video_path, frame_index)
                tag_str = self.tag_image(
                    image,
                    threshold,
                    character_threshold,
                    exclude_tags,
                    replace_underscore,
                    trailing_comma
                )
                results.append(tag_str)
            except Exception as e:
                print(f"[ERROR] {video_path}: {e}")
                results.append("")

        return results


if __name__ == "__main__":

    import os

    model_dir = os.path.join(os.environ['BASE_PATH'],"wd14_tagger/wd14_models")
    model_name = "wd-vit-tagger-v3" ## wd-v1-4-moat-tagger-v2 or wd-vit-tagger-v3 is recommended

    video_list =  ["/home/cho/Code/Make_data_teftef/Result_data/proto_1/00000003.mp4","/home/cho/Code/Make_data_teftef/Result_data/proto_1/00000004.mp4"]
    tagger = WD14VideoTagger(model_dir=model_dir, model_name=model_name)


    tags = tagger.tag_videos(
        video_list,
        threshold=0.4,
        replace_underscore=False
    )

    for path, tag in zip(video_list, tags):
        print(f"{path}: {tag}")