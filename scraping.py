import yt_dlp
import json

def extract_playlist_videos(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
    }

    video_urls = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
        except Exception as e:
            print(f"❌ エラー ({playlist_url}): {e}")
            return []

        if 'entries' in info:
            for entry in info['entries']:
                if entry and 'id' in entry:
                    video_url = f"https://youtu.be/{entry['id']}"
                    video_urls.append(video_url)
        else:
            print(f"⚠️ {playlist_url} の中身が見つかりませんでした。")

    return video_urls

def save_as_json(video_urls, filename='Blue_archive_videos.json'):
    data = {'videos': video_urls}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ {filename} に {len(video_urls)} 件の動画を保存しました。")


if __name__ == "__main__":

    # ✅ ここに複数の再生リストURLを入れる
    playlist_urls = [
        "https://youtube.com/playlist?list=PLYq4yLvct07nrKLfAWL1FtwHCbsUpL9W_",
        "https://youtube.com/playlist?list=PLYq4yLvct07kySJMctdPTYv_SuOZtPgw_",
        "https://youtube.com/playlist?list=PLYq4yLvct07lxOUyeZ0brcOUQdjjLbMfm",
        "https://youtube.com/playlist?list=PLYq4yLvct07kRh0u9huOmJq6Q1HPSIWgy",
        "https://youtube.com/playlist?list=PLYq4yLvct07mCo7eK0P-SxRd1A5VD2KI7",
        "https://youtube.com/playlist?list=PLYq4yLvct07lt3ex9SsiZfeyU-MfZRYFI",
        "https://youtube.com/playlist?list=PLYq4yLvct07m-8T-pJG-2LEPS-46vKENP",
    ]

    all_video_urls = []

    for url in playlist_urls:
        video_urls = extract_playlist_videos(url)
        all_video_urls.extend(video_urls)

    # 重複排除（任意）
    all_video_urls = list(set(all_video_urls))

    # 保存
    save_as_json(all_video_urls)
