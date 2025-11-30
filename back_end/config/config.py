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