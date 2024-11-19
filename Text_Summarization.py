import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer:
    def __init__(self, 
                 model_name="facebook/bart-large-cnn", 
                 max_length=150, 
                 min_length=50, 
                 do_sample=False):
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            # Store configuration
            self.max_length = max_length
            self.min_length = min_length
            self.do_sample = do_sample
        
        except Exception as e:
            print(f"Error initializing summarization model: {e}")
            raise
    
    def summarize(self, text, max_length=None, min_length=None):
        
        # Validate input
        if not text or len(text.strip()) == 0:
            return ""
        
        # Use provided or default lengths
        max_len = max_length or self.max_length
        min_len = min_length or self.min_length
        
        try:
            # Generate summary
            summary = self.summarizer(
                text, 
                max_length=max_len, 
                min_length=min_len, 
                do_sample=self.do_sample
            )
            
            # Return summarized text
            return summary[0]['summary_text'] if summary else ""
        
        except Exception as e:
            print(f"Error during summarization: {e}")
            return ""
    
    def chunk_and_summarize(self, long_text, chunk_size=1000, overlap=100):
        
        # Break text into chunks
        chunks = []
        for i in range(0, len(long_text), chunk_size - overlap):
            chunk = long_text[i:i + chunk_size]
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = [self.summarize(chunk) for chunk in chunks]
        
        # Combine and summarize chunk summaries
        combined_summary = " ".join(chunk_summaries)
        final_summary = self.summarize(combined_summary)
        
        return final_summary

# Example usage
def main():
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Example text
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.
    
    Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Based on long-standing goals in the field, NLP tasks include: machine translation, textual entailment, discourse analysis, text summarization, question answering, and natural language queries to databases.
    """
    
    # Generate summary
    summary = summarizer.summarize(sample_text)
    print("Summary:", summary)
    
    # Demonstrate chunk summarization for longer texts
    long_text = sample_text * 5  # Simulate a longer document
    long_summary = summarizer.chunk_and_summarize(long_text)
    print("\nLong Text Summary:", long_summary)

if __name__ == "__main__":
    main()