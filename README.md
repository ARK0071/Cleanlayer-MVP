# Vendor Name Canonicalization with LangChain

A comprehensive Python pipeline for processing vendor names using semantic embeddings and LLM-powered canonicalization.

## Features

✅ **Semantic Embeddings**: Uses sentence-transformers for vendor name similarity  
✅ **Smart Clustering**: Groups similar vendor names using cosine similarity  
✅ **LLM Integration**: LangChain + OpenAI for canonical name selection  
✅ **Cost Management**: Batching and cost tracking for API efficiency  
✅ **International Support**: 30+ business suffixes (English, German, French, etc.)  
✅ **Production Ready**: Error handling, retry logic, and comprehensive logging  

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=100
BATCH_SIZE=5
BATCH_DELAY=1.0
```

### 3. Basic Usage

```python
from integrated_vendor_processor import ComprehensiveVendorProcessor
import pandas as pd

# Your vendor data
vendor_data = [
    "Microsoft Corporation",
    "Microsoft Corp.",
    "Google LLC", 
    "Google Inc.",
    "Walmart Inc.",
    "Walmart Stores Inc."
]

# Create processor
processor = ComprehensiveVendorProcessor()

# Process vendors
df = pd.DataFrame({'Vendor Name': vendor_data})
df['Cleaned Name'] = df['Vendor Name'].apply(processor.preprocess_vendor_name)

# Generate embeddings and find clusters
embeddings = processor.generate_embeddings(df['Cleaned Name'].tolist())
clusters, _ = processor.find_clusters(embeddings)

# Generate LLM prompts and get canonical names
cluster_prompts = processor.generate_llm_prompts(df, clusters)
processed_prompts = processor.process_all_llm_prompts(cluster_prompts)

# Add canonical names to dataframe
df = processor.add_canonical_names_to_dataframe(df, clusters, processed_prompts)

print(df[['Vendor Name', 'Canonical Name']])
```

## Demo Mode

Run without an API key to see the system working with mock responses:

```bash
python demo_langchain.py
```

## Full Pipeline

Run the complete processing pipeline:

```bash
python integrated_vendor_processor.py
```

## Output Files

The system generates several output files:

- `comprehensive_vendor_results.csv` - Final vendor data with canonical names
- `comprehensive_vendor_results_llm.json` - LLM processing results and costs
- `vendor_cluster_prompts.txt` - Generated prompts for manual review

## API Cost Management

### Batching Strategy
- **Batch Size**: Process 5 prompts at a time (configurable)
- **Delays**: 1 second between batches to avoid rate limits
- **Cost Tracking**: Real-time cost monitoring per request

### Cost Estimates (GPT-3.5-turbo)
- **Small dataset** (100 vendors): ~$0.50-$2.00
- **Medium dataset** (1,000 vendors): ~$5.00-$20.00  
- **Large dataset** (10,000 vendors): ~$50.00-$200.00

*Costs depend on cluster count and prompt complexity*

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `LLM_TEMPERATURE` | `0.1` | Model creativity (0-1) |
| `LLM_MAX_TOKENS` | `100` | Max response length |
| `BATCH_SIZE` | `5` | Prompts per batch |
| `BATCH_DELAY` | `1.0` | Seconds between batches |

### Clustering Parameters

```python
# Adjust these in your code
clusters, similarity_matrix = processor.find_clusters(
    embeddings, 
    k=5,                    # Number of nearest neighbors
    similarity_threshold=0.85  # Minimum similarity for clustering
)
```

## Business Suffixes Supported

The system automatically removes 30+ international business suffixes:

**English**: inc, corp, llc, ltd, plc, lp, llp  
**German**: gmbh, ag, kg, ohg  
**Dutch**: bv, nv  
**French**: sa, sas, sarl, sca, sci, sce  
**Italian**: srl, spa, snc  
**Spanish**: sl, sau  
**Nordic**: oy, ab, as, asa  
**Australian**: pty, pty ltd  
**Asian**: sdn bhd, pte, pte ltd, pvt ltd  

## Production Deployment

### 1. Error Handling
The system includes comprehensive error handling:
- API failures fallback to original names
- Rate limit handling with exponential backoff
- Malformed response detection

### 2. Monitoring
- Real-time cost tracking
- Processing time metrics
- Success/failure rates
- Batch processing status

### 3. Scaling Considerations
- **Memory**: ~7GB RAM for 30K vendors
- **Processing Time**: ~2-5 minutes per 1K vendors
- **API Limits**: Respects OpenAI rate limits

## Integration Examples

### With Pandas DataFrame
```python
# Load your data
df = pd.read_csv('your_vendors.csv')

# Process with the pipeline
processor = ComprehensiveVendorProcessor()
# ... processing steps ...

# Save results
df.to_csv('processed_vendors.csv', index=False)
```

### With Database
```python
import sqlite3

# Load from database
conn = sqlite3.connect('vendors.db')
df = pd.read_sql_query("SELECT vendor_name FROM vendors", conn)

# Process vendors
# ... processing steps ...

# Save back to database
df.to_sql('processed_vendors', conn, if_exists='replace', index=False)
```

### API Endpoint
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
processor = ComprehensiveVendorProcessor()

@app.route('/canonicalize', methods=['POST'])
def canonicalize_vendors():
    vendor_names = request.json['vendors']
    
    # Process vendors
    df = pd.DataFrame({'Vendor Name': vendor_names})
    # ... processing steps ...
    
    return jsonify(df[['Vendor Name', 'Canonical Name']].to_dict('records'))
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade langchain langchain-openai langchain-community
```

**2. API Key Issues**
- Ensure `.env` file is in the same directory
- Check API key format and permissions
- Verify OpenAI account has sufficient credits

**3. Memory Issues**
- Reduce batch size for large datasets
- Process in chunks for 50K+ vendors
- Use FAISS for very large datasets

**4. Rate Limiting**
- Increase `BATCH_DELAY` in `.env`
- Reduce `BATCH_SIZE` 
- Consider OpenAI tier limits

### Performance Optimization

**For Large Datasets (10K+ vendors):**
```python
# Process in chunks
chunk_size = 5000
for i in range(0, len(vendor_data), chunk_size):
    chunk = vendor_data[i:i+chunk_size]
    # Process chunk
```

**For Cost Optimization:**
- Use GPT-3.5-turbo instead of GPT-4
- Optimize prompts for shorter responses
- Pre-filter obvious duplicates

## License

MIT License - feel free to use in your projects!

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues or questions:
- Check the troubleshooting section
- Review the demo scripts
- Open an issue on GitHub 