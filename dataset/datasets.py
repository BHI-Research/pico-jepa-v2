import torch
import torchvision.transforms.v2 as T
import torchcodec
from torch.utils.data import Dataset
import pandas as pd
import os
import random


class VideoDataset(Dataset):
    """
    A unified dataset for loading video clips for various tasks.
    - For pre-training (unlabeled): Returns video clips.
    - For classification (labeled): Returns video clips and their labels.
    - Supports different frame sampling strategies ('random', 'center').
    - Handles videos shorter than the desired clip length by padding.
    """

    def __init__(
        self,
        video_dir,
        csv_file=None,
        frames_per_clip=8,
        target_height=224,
        target_width=224,
        channels=3,
        labeled=False,
        sampling_strategy="random",  # 'random' or 'center'
        csv_path=None,
        indices=None,
    ):
        self.video_dir = video_dir
        if csv_path is not None:
            self.csv_path = csv_path
        elif csv_file is not None:
            self.csv_path = os.path.join(video_dir, csv_file)
        else:
            raise ValueError("VideoDataset requires either csv_file or csv_path.")
        self.channels = channels
        self.labeled = labeled
        self.sampling_strategy = sampling_strategy
        self.frames_per_clip = frames_per_clip
        self.target_height = target_height
        self.target_width = target_width

        # --- CSV Loading ---
        try:
            if self.labeled:
                # Expects 'video_name' and 'label' columns with a header
                self.video_files_df = pd.read_csv(self.csv_path)
                if not {"video_name", "label"}.issubset(self.video_files_df.columns):
                    raise ValueError(
                        "CSV must contain 'video_name' and 'label' columns."
                    )
                self.video_files_df.dropna(subset=["video_name", "label"], inplace=True)
                self.video_files_df["label"] = pd.to_numeric(
                    self.video_files_df["label"]
                )
            else:
                # Expects only video names, no header
                self.video_files_df = pd.read_csv(
                    self.csv_path, header=None, names=["video_name"], usecols=[0]
                )
                self.video_files_df.dropna(subset=["video_name"], inplace=True)
                self.video_files_df = self.video_files_df[
                    self.video_files_df["video_name"].str.strip() != ""
                ]
        except FileNotFoundError:
            print(f"Error: CSV file {self.csv_path} not found!")
            columns = ["video_name", "label"] if self.labeled else ["video_name"]
            self.video_files_df = pd.DataFrame(columns=columns)
        except Exception as e:
            print(f"Error processing CSV {self.csv_path}: {e}")
            columns = ["video_name", "label"] if self.labeled else ["video_name"]
            self.video_files_df = pd.DataFrame(columns=columns)

        if indices is not None:
            self.video_files_df = self.video_files_df.iloc[list(indices)].reset_index(drop=True)

        # --- Transformations ---
        self.transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.Resize((target_height, target_width), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.video_files_df)

    def _get_dummy_data(self):
        """Returns a dummy tensor, and a dummy label if labeled."""
        dummy_tensor = torch.randn(
            self.channels, self.frames_per_clip, self.target_height, self.target_width
        )
        if self.labeled:
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)
        return dummy_tensor

    def __getitem__(self, idx):
        if idx >= len(self.video_files_df):
            print(
                f"Warning: Index {idx} out of bounds for dataset of length {len(self.video_files_df)}."
            )
            return self._get_dummy_data()

        row = self.video_files_df.iloc[idx]
        video_name = row["video_name"]
        label = int(row["label"]) if self.labeled else None
        video_path = os.path.join(self.video_dir, video_name)

        try:
            decoder = torchcodec.decoders.VideoDecoder(video_path)
            meta = decoder.metadata
            total_video_frames = meta.num_frames

            if total_video_frames == 0:
                print(
                    f"Warning: Video {video_path} reported 0 frames. Returning dummy data."
                )
                return self._get_dummy_data()

            # --- Frame Sampling ---
            if total_video_frames >= self.frames_per_clip:
                if self.sampling_strategy == "random":
                    start_frame_index = random.randint(
                        0, total_video_frames - self.frames_per_clip
                    )
                elif self.sampling_strategy == "center":
                    start_frame_index = (total_video_frames - self.frames_per_clip) // 2
                else:  # Default to random
                    start_frame_index = random.randint(
                        0, total_video_frames - self.frames_per_clip
                    )
                frame_indices_to_decode = list(
                    range(start_frame_index, start_frame_index + self.frames_per_clip)
                )
            else:
                # Video is shorter, take all frames (padding will be applied later)
                frame_indices_to_decode = list(range(total_video_frames))

            if not frame_indices_to_decode:
                print(
                    f"Warning: No frame indices to decode for {video_path}. Returning dummy data."
                )
                return self._get_dummy_data()

            frame_batch_obj = decoder.get_frames_at(frame_indices_to_decode)
            clip_tchw_orig_res = frame_batch_obj.data

            # --- Padding ---
            if clip_tchw_orig_res.shape[0] < self.frames_per_clip:
                padding_count = self.frames_per_clip - clip_tchw_orig_res.shape[0]
                if padding_count > 0:
                    if clip_tchw_orig_res.shape[0] == 0:
                        print(
                            f"Warning: Decoded 0 frames for {video_path} even when padding. Returning dummy data."
                        )
                        return self._get_dummy_data()
                    last_frame = clip_tchw_orig_res[
                        -1:, :, :, :
                    ]  # Keep dimensions (1, C, H, W)
                    padding_tensor = last_frame.repeat(padding_count, 1, 1, 1)
                    clip_tchw_orig_res = torch.cat(
                        (clip_tchw_orig_res, padding_tensor), dim=0
                    )

            # Final check on frame count
            if clip_tchw_orig_res.shape[0] != self.frames_per_clip:
                print(
                    f"Warning: Clip from {video_path} after processing has {clip_tchw_orig_res.shape[0]} frames, expected {self.frames_per_clip}. Returning dummy data."
                )
                return self._get_dummy_data()

            # --- Transformation ---
            clip_transformed = self.transform(clip_tchw_orig_res)  # TCHW -> TCHW
            video_tensor = clip_transformed.permute(1, 0, 2, 3)  # TCHW -> CTHW

            if self.labeled:
                return video_tensor, torch.tensor(label, dtype=torch.long)
            else:
                return video_tensor

        except Exception as e:
            error_label = f"(label: {label})" if self.labeled else ""
            print(
                f"Error for video {video_path} {error_label}: {type(e).__name__} - {e}. Returning dummy data."
            )
            return self._get_dummy_data()


# --- NEW: Centralized function for single video processing ---
def load_and_preprocess_single_video(
    video_path: str, config: dict, for_visualization: bool = False
):
    """
    Loads, samples, and preprocesses a single video file using torchcodec.
    This centralized function is used for inference and visualization.

    Args:
        video_path (str): The full path to the video file.
        config (dict): The project configuration dictionary.
        for_visualization (bool): If True, skips normalization to keep original colors.

    Returns:
        A torch.Tensor of shape (C, T, H, W) or None if an error occurs.
    """
    frames_per_clip = config["frames_per_clip"]
    target_height = config["resize_height"]
    target_width = config["resize_width"]

    # Define transform steps, normalization is conditional
    transform_steps = [
        T.ToDtype(torch.float32, scale=True),
        T.Resize((target_height, target_width), antialias=True),
    ]
    if not for_visualization:
        transform_steps.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    transform = T.Compose(transform_steps)

    try:
        decoder = torchcodec.decoders.VideoDecoder(video_path)
        total_frames = decoder.metadata.num_frames
        if total_frames == 0:
            print(f"Warning: Video {video_path} reported 0 frames.")
            return None

        # Use center crop of frames for deterministic inference/visualization
        start_frame = max(0, (total_frames - frames_per_clip) // 2)
        indices = list(range(start_frame, start_frame + frames_per_clip))

        # Ensure indices are within the valid range of the video
        indices = [min(i, total_frames - 1) for i in indices]

        clip_tchw = decoder.get_frames_at(indices).data

        # Pad if necessary
        if clip_tchw.shape[0] < frames_per_clip:
            padding = frames_per_clip - clip_tchw.shape[0]
            if clip_tchw.shape[0] == 0:
                return None
            last_frame = clip_tchw[-1:, ...]
            clip_tchw = torch.cat(
                [clip_tchw, last_frame.repeat(padding, 1, 1, 1)], dim=0
            )

        # Apply transformations
        clip_transformed = transform(clip_tchw)  # TCHW -> TCHW
        return clip_transformed.permute(1, 0, 2, 3)  # TCHW -> CTHW

    except Exception as e:
        print(f"Error processing single video {video_path}: {e}")
        return None


def load_multi_clip_uniform(
    video_path: str, config: dict, num_clips: int = 10, for_visualization: bool = False
):
    """Sample ``num_clips`` clips uniformly spaced across the video.

    Returns a tensor of shape (num_clips, C, T, H, W) ready to feed the
    classifier, or ``None`` if the video can't be decoded. When the video is
    too short to host ``num_clips`` non-overlapping windows, clips overlap or
    repeat; the last frame is padded as in the single-clip case.

    Used by V-JEPA-style multi-clip evaluation: forward each clip through the
    classifier and average the softmax probabilities.
    """
    if num_clips <= 1:
        single = load_and_preprocess_single_video(
            video_path, config, for_visualization=for_visualization
        )
        return single.unsqueeze(0) if single is not None else None

    frames_per_clip = config["frames_per_clip"]
    target_height = config["resize_height"]
    target_width = config["resize_width"]

    transform_steps = [
        T.ToDtype(torch.float32, scale=True),
        T.Resize((target_height, target_width), antialias=True),
    ]
    if not for_visualization:
        transform_steps.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    transform = T.Compose(transform_steps)

    try:
        decoder = torchcodec.decoders.VideoDecoder(video_path)
        total_frames = decoder.metadata.num_frames
        if total_frames == 0:
            print(f"Warning: Video {video_path} reported 0 frames.")
            return None

        if total_frames <= frames_per_clip:
            # Whole video fits in a single clip; replicate it across the multi-clip dim.
            indices = list(range(total_frames))
            clip = decoder.get_frames_at(indices).data
            if clip.shape[0] < frames_per_clip:
                pad = frames_per_clip - clip.shape[0]
                clip = torch.cat([clip, clip[-1:].repeat(pad, 1, 1, 1)], dim=0)
            clip_t = transform(clip).permute(1, 0, 2, 3)
            return clip_t.unsqueeze(0).repeat(num_clips, 1, 1, 1, 1)

        # Place each clip's start frame at uniformly spaced positions.
        max_start = total_frames - frames_per_clip
        if num_clips == 1:
            starts = [max_start // 2]
        else:
            step = max_start / float(num_clips - 1)
            starts = [int(round(i * step)) for i in range(num_clips)]

        clips = []
        for s in starts:
            indices = [min(s + i, total_frames - 1) for i in range(frames_per_clip)]
            clip = decoder.get_frames_at(indices).data
            if clip.shape[0] < frames_per_clip:
                pad = frames_per_clip - clip.shape[0]
                clip = torch.cat([clip, clip[-1:].repeat(pad, 1, 1, 1)], dim=0)
            clip_t = transform(clip).permute(1, 0, 2, 3)  # (C, T, H, W)
            clips.append(clip_t)
        return torch.stack(clips, dim=0)  # (num_clips, C, T, H, W)

    except Exception as e:
        print(f"Error processing multi-clip video {video_path}: {e}")
        return None
