import imageio
import torch
import numpy as np

class VideoCapture:

    @staticmethod
    def load_frames_from_video(video_path, num_frames, sample='rand'):
        try:
            reader = imageio.get_reader(video_path, 'ffmpeg')
        except Exception as e:
            print(f"Warning: Unable to open video file {video_path}: {str(e)}")
            return None, None

        vlen = len(reader)
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = [(intervals[i], intervals[i + 1] - 1) for i in range(len(intervals) - 1)]

        if sample == 'rand':
            frame_idxs = [np.random.randint(start, end) for start, end in ranges]
        else:
            frame_idxs = [(start + end) // 2 for start, end in ranges]

        frames = []
        for idx in frame_idxs:
            frame = reader.get_data(idx)
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # HWC to CHW
            frames.append(frame)

        if len(frames) < num_frames:
            frames += [frames[-1]] * (num_frames - len(frames))

        frames = torch.stack(frames).float() / 255
        return frames, frame_idxs