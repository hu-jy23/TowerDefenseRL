import argparse
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a folder of images into an MP4 video.")
    parser.add_argument(
        "--load-dir",  # 要合成视频的那些图片帧的目录路径
        required=True, 
        type=str, 
        help="Path to the directory containing the image frames (e.g., ./models/date/best_frames)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    frames_dir = args.load_dir

    # 1. 检查输入目录是否存在
    if not os.path.isdir(frames_dir):
        print(f"ERROR: Directory not found at '{frames_dir}'")
        return

    # 2. 自动推导输出文件路径
    # 逻辑：获取 frames_dir 的上一级目录，将视频保存在那里
    # 例如输入: ./models/17.12.2025/best_frames
    # 输出位置: ./models/17.12.2025/best_episode_video.mp4
    
    # os.path.normpath 用于去除路径末尾可能多余的 '/'，确保 dirname 能正确获取父目录
    clean_path = os.path.normpath(frames_dir)
    parent_dir = os.path.dirname(clean_path)
    output_filename = "best_episode_video.mp4"
    output_file = os.path.join(parent_dir, output_filename)

    print(f"--- Configuration ---")
    print(f"Input Directory:  {frames_dir}")
    print(f"Output File:      {output_file}")

    # 3. 获取所有图片并排序
    valid_extensions = ('.png', '.jpg', '.jpeg')
    try:
        images = [
            os.path.join(frames_dir, img) 
            for img in os.listdir(frames_dir) 
            if img.lower().endswith(valid_extensions)
        ]
        # 非常重要：按文件名排序，否则视频会乱序
        images.sort()
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    if not images:
        print(f"Error: No image files found in '{frames_dir}'")
        return

    print(f"Found {len(images)} frames. processing...")

    # 4. 合成视频
    fps = 30  # 播放帧率
    try:
        clip = ImageSequenceClip(images, fps=fps)
        # codec="libx264" 是生成 MP4 的标准编码
        clip.write_videofile(output_file, codec="libx264", logger="bar")
        print(f"\n✅ Success! Video saved to:\n{output_file}")
    except Exception as e:
        print(f"ERROR: Fail to creating video: {e}")

if __name__ == "__main__":
    main()