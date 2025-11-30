from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)
def encode_texts(transcripts):
    """Encode a list of texts into embeddings using the BGEM3FlagModel.
    
    Args:
        transcripts (list of str): List of text strings to encode.
        
    Returns:
        list of list of float: List of embeddings corresponding to each input text.
    """
    embeddings = model.encode(transcripts,
                              batch_size=16,
                              return_dense=True)
    return embeddings.tolist()


