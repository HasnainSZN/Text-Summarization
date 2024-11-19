# Text Summarization Tool

## Overview
A powerful Python-based text summarization tool using transformer models to generate concise summaries of text documents.

## Features
- Transformer-based summarization using BART model
- GPU acceleration support
- Flexible summary length configuration
- Chunk-based summarization for long documents
- Error handling and input validation

## Prerequisites
- Python 3.8+
- PyTorch
- Transformers library


## Configuration Options
- `model_name`: Hugging Face transformer model
- `max_length`: Maximum summary length
- `min_length`: Minimum summary length
- `do_sample`: Enable sampling in text generation

## Supported Models
- Default: facebook/bart-large-cnn
- Other compatible models:
  - google/pegasus-xsum
  - t5-small
  - t5-base

## Performance Considerations
- GPU recommended for faster processing
- Long documents processed in chunks
- Adjust `chunk_size` and `overlap` for optimal results

## Error Handling
- Handles empty input
- Graceful error logging
- Fallback mechanisms

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request


## Acknowledgments
- Hugging Face Transformers
- PyTorch Team

