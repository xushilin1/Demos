import imageio
import argparse
from pathlib import Path
import warnings

# 忽略imageio的无关警告
warnings.filterwarnings('ignore')

def mp4_to_gif_simple(input_path: str, fps: int = 10, loop: int = 0):
    # 构建输出路径
    output_path = input_path.replace('.mp4', '.gif')
    
    # 读取视频
    reader = imageio.get_reader(input_path)
    fps_video = reader.get_meta_data().get('fps', 30)  # 兼容不同视频的元数据
    frame_interval = max(1, int(fps_video / fps))
    
    # 创建GIF写入器，设置循环参数
    # loop=0 表示无限循环，loop=-1 表示不循环，loop=N 表示循环N次
    writer = imageio.get_writer(
        output_path,
        mode='I',
        fps=fps,
        loop=loop  # 核心参数：控制GIF循环播放
    )
    
    # 逐帧写入（按间隔采样）
    frame_count = 0
    for i, frame in enumerate(reader):
        if i % frame_interval == 0:
            writer.append_data(frame)
            frame_count += 1
    
    # 关闭写入器
    writer.close()
    reader.close()
    
    print(f"✅ GIF生成完成：{output_path}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将MP4视频转换为支持循环播放的GIF动图')
    parser.add_argument("--video", type=str, required=True, help="输入MP4视频文件路径")
    parser.add_argument("--fps", type=int, default=10, help="GIF帧率（默认10）")
    
    args = parser.parse_args()
    
    mp4_to_gif_simple(args.video, args.fps)