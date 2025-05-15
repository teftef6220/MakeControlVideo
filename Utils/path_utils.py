from pathlib import Path
import os

def resolve_output_path(dir_path: str):
    """
    path を相対に変換する
    BASE_PATH 環境変数を使用して、出力先のディレクトリが BASE_PATH の中にあるか確認する
    もし BASE_PATH が存在しない場合は、ValueError を投げる
    もし dir_path が絶対パスの場合は、BASE_PATH からの相対パスに変換する
    もし dir_path が相対パスの場合は、BASE_PATH からの相対パスに変換する

    EN
    convert the path to relative
    use the BASE_PATH environment variable to check if the output directory is under BASE_PATH
    if BASE_PATH does not exist, raise ValueError
    if dir_path is an absolute path, convert it to a relative path from BASE_PATH
    if dir_path is a relative path, convert it to a relative path from BASE_PATH
    """
    base_path = Path(os.environ['BASE_PATH']).resolve()
    output_path = Path(dir_path)

    if not base_path.exists():
        raise ValueError(f"BASE_PATH '{base_path}' does not exist.")

    if output_path.is_absolute():
        output_path = output_path.resolve()
        try:
            # BASE_PATH の中にあるか確認
            return output_path.relative_to(base_path)
        except ValueError:
            raise ValueError(f"dir_path '{output_path}' is not under BASE_PATH '{base_path}'")
    else:
        # 相対パスとして解釈されるが、base_path からの相対かチェック
        combined = (base_path / output_path).resolve()
        try:
            _ = combined.relative_to(base_path)
            return output_path  # OK、すでに base_path からの相対パス
        except ValueError:
            raise ValueError(f"Relative path '{output_path}' escapes BASE_PATH '{base_path}'")



def collect_video_paths(dir_path, extensions=None):
    """
    指定されたディレクトリ内の動画ファイルのパスを収集し、listで返す
    Collect video file paths in the specified directory and return them in JSON format
    """
    if extensions is None:
        extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v']

    base = Path(dir_path).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a valid directory.")

    return [
        str(p.resolve())
        for p in base.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ]