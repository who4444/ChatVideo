import logging
import os 
from pathlib import Path
from dataclasses import dataclass
import gc
import json
import pickle
import re
import time
from typing import List, Tuple, Optional, Dict, Any

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset 
import open_clip

import numpy as np
from PIL import Image
from pymilvus import MilvusClient, DataType

from tqdm import tqdm

logger = logging.getLogger(__name__)

class Config:
    # model configs
    CLIP_MODEL_NAME = 'ViT-H-14-quickgelu'
    CLIP_PRETRAINED = 'dfn5b'

    # processing:
    BATCH_SIZE = 64
    WORKER_NUM = 4
    BATCH_SAVE_SIZE = 10


    # Paths
    KEYFRAMES_DIR: str = ""
    OUTPUT_DIR: str = ""
    MILVUS_URI: str = ""
    CLIP_COLLECTION_NAME: str = "clip_features"
    
    # Memory management
    MEMORY_THRESHOLD: float = 0.85
    CHECKPOINT_INTERVAL: int = 5
    MAX_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    MIN_CROP_SIZE: int = 32
    
    CLIP_VECTOR_DIM = 1024


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
                image = image.convert('RGB')  # Fix: use parentheses, not brackets
            
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
    
            # Combined pattern for K- and L-series videos
            pattern = r"^(?P<video>[KL]\d+_V\d+)_frame_(?P<frame>\d+)_timestamp_(?P<ts>[\d.]+)s$"
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
    
class MemoryManager:
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    @staticmethod
    def cleanup_memory():
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def check_memory_threshold(threshold: float = 0.85) -> bool:
        """Check if memory usage exceeds threshold"""
        return MemoryManager.get_memory_usage() > threshold
    
class Encoder:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_ids = list(range(torch.cuda.device_count()))
        
        logger.info(f"Initializing Encoder on {self.device} with {len(self.device_ids)} GPUs")        
        
        # Load CLIP model
        self.load_models()

        # Load checkpoint
        self.processed_videos, loaded_stats = self.checkpoint_manager.load_checkpoint()
        if loaded_stats.videos_processed > 0:
            self.stats = loaded_stats

        self.clip_coll = self.init_milvus()
        
    def load_models(self):
        """Load CLIP model with error handling"""
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=self.config.CLIP_MODEL_NAME,
            pretrained=self.config.CLIP_PRETRAINED,
            precision='fp16'
        )
        self.clip_model = self.clip_model.to(self.device)
        if len(self.device_ids) > 1:
            self.clip_model = nn.DataParallel(self.clip_model, device_ids=self.device_ids)
        self.clip_model.eval()
             
    def init_milvus(self):
        """Setup Milvus collection with proper schema"""
        try:
            self.client = MilvusClient(self.config.MILVUS_URI)
            
            # Create CLIP collection if it doesn't exist
            clip_coll = self.create_collection(self.config.CLIP_COLLECTION_NAME)
            
            return clip_coll
            
        except Exception as e:
            logger.error(f"Failed to setup Milvus: {e}")
            raise

    def create_collection(self, collection_name: str):
        """Create a Milvus collection with proper schema"""
        if self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return collection_name
            
        vector_dim = self.config.CLIP_VECTOR_DIM
            
        # Create schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field(field_name="video_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="frame_num", datatype=DataType.INT64)
        schema.add_field(field_name="timestamp", datatype=DataType.FLOAT)
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema
        )
        
        # Create index
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name='vector',
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )
            
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        
        logger.info(f"Created collection: {collection_name}")
        return collection_name

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
                if batch_count % 10 == 0:
                    MemoryManager.cleanup_memory()
            
            self.stats.frames_processed += frames_in_video
            return clip_entities, None
            
        except Exception as e:
            error_msg = f"Error processing video {video_name}: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def get_video_folders(self, keyframes_dir: str = None):
        if keyframes_dir is None:
            keyframes_dir = self.config.KEYFRAMES_DIR
            
        video_folders = {}
        img_ext = ('.jpg', '.jpeg', '.png')
    
        for root, dirs, files in os.walk(keyframes_dir):
            # Look for image files in this directory
            image_files = [
                os.path.join(root, f)
                for f in files
                if f.lower().endswith(img_ext)
            ]
            if image_files:
                video_name = os.path.basename(root)  # e.g. "L21_V001"
                video_folders[video_name] = image_files
        return video_folders
            
    def _save_progress(self, clip_entities: List[Dict]) -> bool:
        """Save entities to Milvus with better error handling"""
        if not clip_entities:
            return True
            
        try:
            res = self.client.insert(self.config.CLIP_COLLECTION_NAME, clip_entities)
            logger.info(f"Successfully inserted {len(clip_entities)} entities to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert entities to Milvus: {e}")
            
            # Save to backup files
            backup_dir = Path(self.config.OUTPUT_DIR) / "backup"
            backup_dir.mkdir(exist_ok=True)
            
            backup_file = backup_dir / f"clip_backup_{int(time.time())}.pkl"
            try:
                with open(backup_file, 'wb') as f:
                    pickle.dump(clip_entities, f)
                logger.info(f"CLIP entities saved to backup: {backup_file}")
            except Exception as backup_error:
                logger.error(f"Failed to save backup: {backup_error}")
                
            return False
    
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
                        
                        logger.info(f"✅ [{i+1}/{len(remaining_videos)}] {video_name}: {len(clip_entities)} CLIP entities")
                    else:
                        self.stats.videos_failed += 1
                        errors.append({'video': video_name, 'error': error})
                        logger.error(f"❌ [{i+1}/{len(remaining_videos)}] {video_name}: {error}")
                    
                    # Save progress when we have enough entities or at checkpoint interval
                    should_save = (
                        entities_since_last_save >= self.config.BATCH_SAVE_SIZE or
                        (i + 1) % self.config.CHECKPOINT_INTERVAL == 0 or
                        i == len(remaining_videos) - 1  # Last video
                    )
                    
                    if should_save and all_clip_entities:
                        success = self._save_progress(all_clip_entities)
                        if success:
                            all_clip_entities = []  # Clear to save memory
                            entities_since_last_save = 0
                        
                        # Save checkpoint
                        self.checkpoint_manager.save_checkpoint(
                            self.processed_videos, self.stats
                        )
                    
                    # Memory management
                    if MemoryManager.check_memory_threshold(self.config.MEMORY_THRESHOLD):
                        MemoryManager.cleanup_memory()
            
            # Final save if there are remaining entities
            if all_clip_entities:
                self._save_progress(all_clip_entities)
            
            # Final checkpoint
            self.checkpoint_manager.save_checkpoint(self.processed_videos, self.stats)
            
            # Summary
            summary = {
                'success': True,
                'stats': self.stats.get_summary(),
                'errors': errors,
                'total_entities_inserted': self.stats.total_entities
            }
            
            logger.info(f"Processing complete: {summary['stats']}")
            return summary
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'success': False, 'error': str(e)}
