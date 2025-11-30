import logging
import os 
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from back_end.config import Config

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset 
import open_clip

import numpy as np
from PIL import Image
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)

class FrameDataset(Dataset):
    def __init__(self, image_files, preprocess, max_image_size: Tuple[int, int] = (512, 512)):
        self.image_files = image_files
        self.preprocess = preprocess
        self.max_image_size = max_image_size
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx):
       
        img_path = self.image_files[idx]
        
        try:
            # Open and validate image
            image = Image.open(img_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if needed
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Preprocess for CLIP
            tensor = self.preprocess(image)
            
            # Extract metadata
            video_name = Path(img_path).parent.name
            video_id, frame_num, timestamp = self._parse_filename(Path(img_path).name, video_name)
            
            metadata = {
                "video_name": video_id,
                "frame_num": frame_num,
                "timestamp": timestamp,
            }
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return None, None
    
    def _parse_filename(self, filename: str, fallback_video_name: str) -> tuple:
        try:
            base, _ = os.path.splitext(filename)
    
            pattern = r""
            match = re.match(pattern, base)
    
            if match:
                video_name = match.group("video")
                frame_num = int(match.group("frame"))
                timestamp = float(match.group("ts"))
                return video_name, frame_num, timestamp
            else:
                # Fallback if the pattern doesn't match
                logger.warning(f"Filename {filename} did not match expected format.")
                return fallback_video_name, 0, 0.0
        except Exception as e:
            logger.warning(f"Could not parse filename {filename}: {e}")
            return fallback_video_name, 0, 0.0
    
    def get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.image_files)):
            try:
                tensor, metadata = self.__getitem__(idx)
                if tensor is not None and metadata is not None:
                    valid_indices.append(idx)
            except Exception:
                continue
        return valid_indices
    
    @staticmethod
    def collate_fn(batch):
        tensors = []
        metadata_list = []
        
        for tensor, metadata in batch:
            if tensor is not None and metadata is not None:
                tensors.append(tensor)
                metadata_list.append(metadata)
        
        return tensors, metadata_list
    
class FrameEncoder:
    def __init__(self, config: Config = None):
        self.config = config 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.config.CLIP_MODEL_NAME,
            pretrained=self.config.CLIP_PRETRAINED,
            precision='fp16'
        )
        self.clip_model = self.clip_model.to(self.device)
        if len(self.device_ids) > 1:
            self.clip_model = nn.DataParallel(self.clip_model, device_ids=self.device_ids)
        self.clip_model.eval()
             
    def process_single_video(self, image_files) -> Tuple[List[Dict], Optional[str]]:
        """Process a single video folder with enhanced error handling"""
        if not image_files:
            return [], "No image files provided"
            
        video_name = Path(image_files[0]).parent.name
        logger.info(f"Processing video: {video_name} with {len(image_files)} frames")
               
        clip_entities = []
        frames_in_video = 0
        
        try:
            dataset = FrameDataset(
                image_files=image_files,
                preprocess=self.preprocess
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                num_workers=self.config.WORKER_NUM,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                collate_fn=FrameDataset.collate_fn
            )
            
            batch_count = 0
            for batch_tensors, batch_metadata in dataloader:
                valid_indices = [i for i, t in enumerate(batch_tensors) if t is not None]
                if not valid_indices:
                    continue
            
                valid_tensors = torch.stack([batch_tensors[i] for i in valid_indices]).to(self.device, dtype=torch.float16)            
                valid_metadata = [batch_metadata[i] for i in valid_indices]
                
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    if isinstance(self.clip_model, nn.DataParallel):
                        clip_features = self.clip_model.module.encode_image(valid_tensors)
                    else:
                        clip_features = self.clip_model.encode_image(valid_tensors)
                    
                    clip_features = clip_features.cpu().numpy()
                    clip_features /= np.linalg.norm(clip_features, axis=1, keepdims=True)
                
                # Process each frame
                for clip_emb, metadata in zip(clip_features, valid_metadata):
                    if metadata is None:  # Skip if metadata is None
                        continue
                    clip_entities.append({
                        "vector": clip_emb.tolist(),
                        "video_name": metadata["video_name"],
                        "frame_num": metadata["frame_num"],
                        "timestamp": metadata["timestamp"],
                    })
                    frames_in_video += 1
                
                # Clean up batch data
                del valid_tensors, batch_tensors, batch_metadata
                
                batch_count += 1
                # Only cleanup memory every 10 batches to reduce overhead

            self.stats.frames_processed += frames_in_video
            return clip_entities, None
            
        except Exception as e:
            error_msg = f"Error processing video {video_name}: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    # def get_video_folders(self, keyframes_dir: str = None):
    #     if keyframes_dir is None:
    #         keyframes_dir = self.config.KEYFRAMES_DIR
            
    #     video_folders = {}
    #     img_ext = ('.jpg', '.jpeg', '.png')
    
    #     for root, dirs, files in os.walk(keyframes_dir):
    #         # Look for image files in this directory
    #         image_files = [
    #             os.path.join(root, f)
    #             for f in files
    #             if f.lower().endswith(img_ext)
    #         ]
    #         if image_files:
    #             video_name = os.path.basename(root)  # e.g. "L21_V001"
    #             video_folders[video_name] = image_files
    #     return video_folders
            

    def process_all_videos(self, keyframes_dir: str = None) -> Dict[str, Any]:
        """Process all videos with checkpointing and error recovery"""
        if keyframes_dir is None:
            keyframes_dir = self.config.KEYFRAMES_DIR
        
        # Get video folders and their image files
        video_folders = self.get_video_folders(keyframes_dir)
        
        if not video_folders:
            logger.error("No video folders found")
            return {'success': False, 'message': 'No video folders found'}
        
        # Filter out already processed videos
        remaining_videos = {
            video_name: image_files
            for video_name, image_files in video_folders.items()
            if video_name not in self.processed_videos
        }
        
        if self.config.MAX_VIDEOS:
            remaining_videos = dict(list(remaining_videos.items())[:self.config.MAX_VIDEOS])
        
        logger.info(f"Processing {len(remaining_videos)} videos (skipped {len(video_folders) - len(remaining_videos)} already processed)")
        
        all_clip_entities = []
        errors = []
        entities_since_last_save = 0
        
        try:
            with tqdm(remaining_videos.items(), desc="Processing videos") as pbar:
                for i, (video_name, image_files) in enumerate(pbar):
                    pbar.set_postfix({'video': video_name[:20]})

                    # Process video
                    clip_entities, error = self.process_single_video(image_files)
                    
                    if error is None and clip_entities:
                        all_clip_entities.extend(clip_entities)
                        entities_since_last_save += len(clip_entities)
                        
                        self.stats.videos_processed += 1
                        self.stats.total_entities += len(clip_entities)
                        self.processed_videos.append(video_name)
                        
                        logger.info(f"[{i+1}/{len(remaining_videos)}] {video_name}: {len(clip_entities)} CLIP entities")
                    else:
                        self.stats.videos_failed += 1
                        errors.append({'video': video_name, 'error': error})
                        logger.error(f"[{i+1}/{len(remaining_videos)}] {video_name}: {error}")
                    
                    # Save progress when we have enough entities or at checkpoint interval
                    should_save = (
                        entities_since_last_save >= self.config.BATCH_SAVE_SIZE or
                        i == len(remaining_videos) - 1  # Last video
                    )
                    
                    if should_save and all_clip_entities:
                        success = self._save_progress(all_clip_entities)
                        if success:
                            all_clip_entities = []  # Clear to save memory
                            entities_since_last_save = 0
                                 
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'success': False, 'error': str(e)}
