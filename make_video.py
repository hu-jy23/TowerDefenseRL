from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os

# 这里填你刚才日志里显示的那个保存路径
frames_dir = "./models/18.12.2025_00.27/best_frames/"
output_file = "./best_replay.mp4"
fps = 30  # 播放速度

# 1. 获取所有图片文件并按文件名排序
images = sorted([
    os.path.join(frames_dir, img) 
    for img in os.listdir(frames_dir) 
    if img.endswith(".png")
])

print(f"Found {len(images)} frames, creating video...")

# 2. 合成视频
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile(output_file, codec="libx264")

print(f"Done! Video saved to: {output_file}")