
# Swift OCR: Multi-Method LLM Powered Fast OCR ⚡

## 🌟 Features

- **Multiple OCR Methods**: Choose from GPT-4 Vision, Tesseract, or Hybrid approaches for different accuracy and speed requirements
- **Flexible Input Options**: Accepts PDF files via direct upload, binary upload, or by specifying a URL
- **Advanced OCR Processing**: 
  - **GPT-4 Vision**: Utilizes OpenAI's GPT-4 Vision model for highest accuracy text extraction
  - **Tesseract OCR**: Fast, open-source OCR for quick processing
  - **Hybrid Method**: Combines both GPT-4 Vision and Tesseract results for comprehensive analysis
- **Performance Optimizations**:
  - **Parallel PDF Conversion**: Converts PDF pages to images concurrently using multiprocessing
  - **Batch Processing**: Processes multiple images in batches to maximize throughput
  - **Retry Mechanism with Exponential Backoff**: Ensures resilience against transient failures and API rate limits
- **Structured Output**: Extracted text is formatted using Markdown for readability and consistency
- **Robust Error Handling**: Comprehensive logging and exception handling for reliable operations
- **Scalable Architecture**: Asynchronous processing enables handling multiple requests efficiently
- **API Authentication**: Secure API key-based authentication for all endpoints

## 📹 Demo

https://github.com/user-attachments/assets/6b39f3ea-248e-4c29-ac2e-b57de64d5d65

*Demo video showcasing the conversion of NASA's Apollo 17 flight documents, which include unorganized, horizontally and vertically oriented pages, into well-structured Markdown format without any issues.*

## 🔧 Available OCR Methods

### 1. GPT-4 Vision OCR (Default)
- **Endpoint**: `POST /ocr`
- **Best for**: Highest accuracy, complex layouts, tables, and mixed content
- **Features**: Markdown formatting, table structure preservation, image descriptions

### 2. Tesseract OCR
- **Endpoint**: `POST /ocr/tesseract`  
- **Best for**: Fast processing, simple text documents, cost-effective solutions
- **Features**: Open-source, quick processing, good for standard documents

### 3. Hybrid OCR
- **Endpoint**: `POST /ocr/hybrid`
- **Best for**: Comprehensive analysis, comparing results, maximum coverage
- **Features**: Combines GPT-4 Vision and Tesseract results in a single response

### 4. List Available Methods
- **Endpoint**: `GET /ocr/methods`
- **Purpose**: Returns all available OCR methods and their descriptions

## Cost Comparison and Value Proposition

Our solution offers an optimal balance of affordability, accuracy, and advanced features:

### Cost Breakdown (GPT-4 Vision)
- Average token usage per image: ~1200
- Total tokens per page (including prompt): ~1500
- [GPT4O] Input token cost: $5 per million tokens
- [GPT4O] Output token cost: $15 per million tokens

For 1000 documents:
- Estimated total cost: $15

#### Cost Optimization Options
1. **Tesseract OCR**: Free open-source option for basic text extraction
2. **Hybrid Approach**: Compare results between methods to choose the best for your use case
3. **GPT4 mini**: Reduces cost to ~$8 per 1000 documents
4. **Batch API**: Further reduces cost to ~$4 per 1000 documents

#### Market Comparison
This solution is significantly more affordable than alternatives:
- Our cost: $15 per 1000 documents (GPT-4 Vision)
- CloudConvert: ~$30 per 1000 documents (PDFTron mode, 4 credits required)
- Tesseract option: Free

While cost-effectiveness is a major advantage, our solution also provides:
- Superior accuracy and consistency (GPT-4 Vision)
- Precise table generation
- Output in easily editable markdown format
- Multiple processing options for different needs

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yigitkonur/llm-openai-ocr.git
   cd llm-openai-ocr
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**

   Create a `.env` file in the root directory and add the following variables:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   API_KEY=your-secret-api-key-here
   MODEL_NAME=gpt-4-vision-preview  # Optional: Default is "gpt-4-vision-preview"
   BATCH_SIZE=1  # Optional: Default is 1
   MAX_CONCURRENT_OCR_REQUESTS=5  # Optional: Default is 5
   MAX_CONCURRENT_PDF_CONVERSION=4  # Optional: Default is 4
   ```

   > **Note:** Replace `your_openai_api_key` with your actual OpenAI API key and set a secure API key for authentication.

4. **Run the Application**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 3000
   ```

   The API will be available at `http://localhost:3000` with interactive documentation at the root URL.

## 🎯 Usage

### Authentication

All endpoints require an API key passed via the `X-API-Key` header:

```bash
-H "X-API-Key: your-secret-api-key-here"
```

### API Endpoints

#### 1. GPT-4 Vision OCR (Highest Accuracy)

**POST** `/ocr`

```bash
# Upload PDF file
curl -X POST "http://localhost:3000/ocr" \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "file=@/path/to/your/document.pdf"

# Process PDF from URL
curl -X POST "http://localhost:3000/ocr" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf"}'

# Direct binary upload
curl -X POST "http://localhost:3000/ocr" \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/pdf" \
  --data-binary @document.pdf
```

#### 2. Tesseract OCR (Fast & Free)

**POST** `/ocr/tesseract`

```bash
# Upload PDF file
curl -X POST "http://localhost:3000/ocr/tesseract" \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "file=@/path/to/your/document.pdf"
```

#### 3. Hybrid OCR (Comprehensive)

**POST** `/ocr/hybrid`

```bash
# Upload PDF file
curl -X POST "http://localhost:3000/ocr/hybrid" \
  -H "X-API-Key: your-secret-api-key-here" \
  -F "file=@/path/to/your/document.pdf"
```

#### 4. List Available Methods

**GET** `/ocr/methods`

```bash
curl -X GET "http://localhost:3000/ocr/methods"
```

### Input Methods

All OCR endpoints support three input methods:

1. **File Upload**: Use multipart form data with a `file` parameter
2. **URL**: Provide a JSON object with a `url` field pointing to a PDF
3. **Binary Upload**: Send PDF data directly with `Content-Type: application/pdf`

### Response Format

All OCR endpoints return the same response format:

```json
{
  "text": "Extracted and formatted text from the PDF in Markdown format."
}
```

#### Error Responses

- `400 Bad Request`: Invalid input parameters or file format
- `401 Unauthorized`: Missing API key
- `403 Forbidden`: Invalid API key
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Processing errors

## 🧰 Configuration

All configurations are managed via environment variables:

### Required Variables

- **OPENAI_API_KEY**: Your OpenAI API key (required for GPT-4 Vision)
- **API_KEY**: Your custom API key for authentication

### Optional Variables

- **MODEL_NAME**: OpenAI model to use (default: "gpt-4-vision-preview")
- **BATCH_SIZE**: Number of images to process per OCR request (default: 1)
- **MAX_CONCURRENT_OCR_REQUESTS**: Maximum concurrent OCR requests (default: 5)
- **MAX_CONCURRENT_PDF_CONVERSION**: Maximum concurrent PDF conversions (default: 4)

### CORS Configuration

The API is configured with CORS to allow requests from any origin. In production, update the `allow_origins` setting in the FastAPI CORS middleware to restrict access to specific domains.

## 🔧 Method Selection Guide

| Method | Speed | Accuracy | Cost | Best For |
|--------|-------|----------|------|----------|
| **Tesseract** | ⚡⚡⚡ | ⭐⭐ | Free | Simple text documents, fast processing |
| **GPT-4 Vision** | ⚡ | ⭐⭐⭐⭐⭐ | $$ | Complex layouts, tables, mixed content |
| **Hybrid** | ⚡ | ⭐⭐⭐⭐ | $$ | Comprehensive analysis, result comparison |

## 📊 Performance Features

- **Multiprocessing PDF Conversion**: Parallel page rendering for faster processing
- **Asynchronous Processing**: Handle multiple requests simultaneously
- **Exponential Backoff**: Automatic retry with intelligent delays
- **Batch Processing**: Process multiple images efficiently
- **Memory Management**: Temporary file cleanup and optimized resource usage

## 📜 License

Please note that PyMuPDF requires changing the license to GNU AGPL v3.0. You can fork this project, implement pdf2image, and use it freely. While I don't have any particular interest in licensing, I am legally obligated to add this information.

GNU AFFERO GENERAL PUBLIC LICENSE
Version 3, 19 November 2007

Copyright (C) 2024 Yiğit Konur

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
