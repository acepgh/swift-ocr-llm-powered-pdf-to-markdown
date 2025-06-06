import asyncio
import base64
import io
import logging
import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, HttpUrl, ValidationError
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Configuration and Initialization
# ----------------------------

load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4-vision-preview")
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-here")  # Replace with your desired API key

    # More robust environment variable parsing
    @staticmethod
    def parse_int_env(key: str, default: int) -> int:
        try:
            value = os.getenv(key)
            if value is None:
                return default
            value = value.split('#')[0].strip()
            return int(value)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid value for {key}, using default: {default}")
            return default

    # Use the new parsing method
    BATCH_SIZE: int = parse_int_env("BATCH_SIZE", 1)
    MAX_CONCURRENT_OCR_REQUESTS: int = parse_int_env("MAX_CONCURRENT_OCR_REQUESTS", 5)
    MAX_CONCURRENT_PDF_CONVERSION: int = parse_int_env("MAX_CONCURRENT_PDF_CONVERSION", 4)

    @classmethod
    def validate(cls):
        logger.info(f"BATCH_SIZE env value: '{os.getenv('BATCH_SIZE')}'")

        missing = [
            var
            for var in ["OPENAI_API_KEY"]
            if not getattr(cls, var)
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


Settings.validate()

# Initialize OpenAI client
try:
    openai_client = AsyncOpenAI(
        api_key=Settings.OPENAI_API_KEY,
        timeout=30.0,
    )
    # Test the client configuration
    logger.info(f"Initializing OpenAI client")
    logger.info(f"Using model: {Settings.MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

# Initialize API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=401, detail="API Key header is missing")
    if api_key_header != Settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key_header

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI Application
app = FastAPI(
    title="Swift OCR API",
    description="""
    🚀 PDF OCR API that converts documents to text using multiple OCR methods:

    ## Features
    * GPT-4 Vision OCR (Most accurate)
    * Tesseract OCR (Fast, open-source)
    * Hybrid approach (Combines both methods)

    ## Authentication
    All endpoints require an API key passed via the `X-API-Key` header.

    ## Input Methods
    * Upload PDF file directly
    * Provide PDF URL
    """,
    version="1.0.0",
    docs_url="/",
    redoc_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ----------------------------
# Logging Configuration
# ----------------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# ----------------------------
# Exception Handlers
# ----------------------------

@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(ValidationError)
async def handle_validation_exception(request: Request, exc: ValidationError):
    logger.error(f"ValidationError: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(Exception)
async def handle_unhandled_exception(request: Request, exc: Exception):
    logger.exception(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# ----------------------------
# Pydantic Models
# ----------------------------

class OCRRequest(BaseModel):
    url: Optional[HttpUrl] = None

class OCRResponse(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": """# Document Title

## Section 1
This is a sample extracted text with **bold** and *italic* formatting.

## Section 2
[Image: A diagram showing workflow steps]

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |"""
            }
        }

# ----------------------------
# Utility Functions
# ----------------------------

def download_pdf(url: str) -> bytes:
    """
    Download a PDF file from the specified URL.

    Args:
        url (str): The URL of the PDF.

    Returns:
        bytes: The content of the PDF.

    Raises:
        HTTPException: If the download fails or the content is not a PDF.
    """
    try:
        response = requests.get(url, timeout=60)  # Increased timeout to 60 seconds
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type:
            logger.warning(f"Invalid content type: {content_type} for URL: {url}")
            raise HTTPException(
                status_code=400, detail="URL does not point to a valid PDF file."
            )
        logger.info(f"Downloaded PDF from {url}, size: {len(response.content)} bytes.")
        return response.content
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while downloading PDF from {url}.")
        raise HTTPException(
            status_code=504, detail="Timeout occurred while downloading the PDF."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while downloading PDF from {url}: {e}")
        raise HTTPException(
            status_code=400, detail=f"HTTP error occurred: {e}"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while downloading PDF from {url}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to download PDF: {e}"
        )

def convert_page_to_image(args: Tuple[str, int, int]) -> Tuple[int, bytes]:
    """
    Convert a single PDF page to PNG image bytes using PyMuPDF.

    Args:
        args (Tuple[str, int, int]): A tuple containing:
            - pdf_path (str): Path to the PDF file.
            - page_num (int): Page number to convert (0-based).
            - zoom (int): Zoom factor for rendering.

    Returns:
        Tuple[int, bytes]: A tuple of page number and image bytes.

    Raises:
        Exception: If rendering fails.
    """
    pdf_path, page_num, zoom = args
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        image_bytes = pix.tobytes("png")
        logger.debug(
            f"Rendered page {page_num + 1}/{doc.page_count}, size: {len(image_bytes)} bytes."
        )
        doc.close()
        return (page_num + 1, image_bytes)  # Page numbers start at 1
    except Exception as e:
        logger.error(f"Error rendering page {page_num + 1}: {e}")
        raise

def convert_pdf_to_images_pymupdf(pdf_path: str, zoom: int = 2) -> List[Tuple[int, bytes]]:
    """
    Convert a PDF file to a list of PNG image bytes using PyMuPDF with multiprocessing.

    Args:
        pdf_path (str): Path to the PDF file.
        zoom (int): Zoom factor for rendering.

    Returns:
        List[Tuple[int, bytes]]: List of tuples containing page number and PNG image bytes.

    Raises:
        HTTPException: If conversion fails.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        logger.info(f"PDF loaded with {page_count} pages.")

        # Prepare arguments for each page
        args_list = [(pdf_path, i, zoom) for i in range(page_count)]

        image_bytes_list: List[Tuple[int, bytes]] = []  # List of (page_num, image_bytes)

        with ProcessPoolExecutor(max_workers=Settings.MAX_CONCURRENT_PDF_CONVERSION) as executor:
            # Submit all tasks
            future_to_page = {
                executor.submit(convert_page_to_image, args): args[1]
                for args in args_list
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, image_bytes = future.result()
                    image_bytes_list.append((page_num_result, image_bytes))
                except Exception as e:
                    logger.error(f"Failed to convert page {page_num + 1}: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to convert page {page_num + 1} to image.",
                    )

        # Sort the list by page number to maintain order
        image_bytes_list.sort(key=lambda x: x[0])

        logger.info(f"Converted PDF to {len(image_bytes_list)} images using PyMuPDF.")
        return image_bytes_list

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error converting PDF to images with PyMuPDF: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to convert PDF to images: {e}"
        )

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to a base64 data URL.

    Args:
        image_bytes (bytes): The image content.

    Returns:
        str: Base64 encoded data URL.

    Raises:
        HTTPException: If encoding fails.
    """
    try:
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_str}"
        logger.debug(
            f"Encoded image to base64 data URL, length: {len(data_url)} characters."
        )
        return data_url
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to encode image to base64."
        )

def encode_images(image_bytes_list: List[Tuple[int, bytes]]) -> List[Tuple[int, str]]:
    """
    Encode a list of image bytes to base64 data URLs along with their page numbers.

    Args:
        image_bytes_list (List[Tuple[int, bytes]]): List of tuples containing page numbers and image bytes.

    Returns:
        List[Tuple[int, str]]: List of tuples containing page numbers and base64-encoded image URLs.
    """
    encoded_urls = [(page_num, encode_image_to_base64(img_bytes)) for page_num, img_bytes in image_bytes_list]
    logger.info(f"Encoded {len(encoded_urls)} images to base64 data URLs.")
    return encoded_urls

def create_batches(items: List[Tuple[int, str]], batch_size: int) -> List[List[Tuple[int, str]]]:
    """
    Split a list of items into batches.

    Args:
        items (List[Tuple[int, str]]): The list of tuples containing page numbers and image URLs.
        batch_size (int): The maximum size of each batch.

    Returns:
        List[List[Tuple[int, str]]]: A list of batches.
    """
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    logger.info(
        f"Divided images into {len(batches)} batches of up to {batch_size} images each."
    )
    return batches

async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 10,
    base_delay: int = 1,
    max_delay: int = 120,
    *args,
    **kwargs
) -> Any:
    """
    Retry a coroutine function with exponential backoff.

    Args:
        func (Callable): The coroutine function to retry.
        max_retries (int): Maximum number of retries.
        base_delay (int): Initial delay in seconds.
        max_delay (int): Maximum delay in seconds.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: The result of the function if successful.

    Raises:
        HTTPException: If all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except HTTPException as he:
            if he.status_code == 429:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"Rate limit encountered. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"HTTPException during retry: {he.detail}")
                raise
        except asyncio.TimeoutError:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                f"Timeout. Retrying in {delay} seconds... (Attempt {attempt}/{max_retries})"
            )
            await asyncio.sleep(delay)
        except Exception as e:
            logger.exception(f"Unexpected error during retry: {e}")
            raise HTTPException(
                status_code=500,
                detail="An unexpected error occurred during processing.",
            )
    logger.error("Exceeded maximum retries.")
    raise HTTPException(
        status_code=429, detail="Maximum retry attempts exceeded."
    )

# ----------------------------
# OCR Service
# ----------------------------

class OCRService:
    def __init__(self):
        try:
            self.client = AsyncOpenAI(
                api_key=Settings.OPENAI_API_KEY,
                timeout=120.0,  # Increased timeout to 2 minutes
            )
        except Exception as e:
            logger.exception(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    async def perform_ocr_on_batch(self, image_batch: List[Tuple[int, str]]) -> str:
        """
        Perform OCR on a batch of images using OpenAI's API with retry logic.

        Args:
            image_batch (List[Tuple[int, str]]): List of tuples containing page numbers and base64-encoded image URLs.

        Returns:
            str: Extracted text.

        Raises:
            HTTPException: If OCR fails after retries.
        """
        async def ocr_request():
            try:
                messages = self.build_ocr_messages(image_batch)
                logger.info(
                    f"Sending OCR request to OpenAI with {len(image_batch)} images."
                )
                response = await self.client.chat.completions.create(
                    model=Settings.MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return self.extract_text_from_response(response)
            except OpenAIError as e:
                if "rate limit" in str(e).lower():
                    raise HTTPException(
                        status_code=429, detail="Rate limit exceeded."
                    )
                else:
                    logger.error(f"OpenAI API error: {e}")
                    raise HTTPException(
                        status_code=502,
                        detail=f"OCR processing failed: {e}",
                    )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="Timeout occurred while communicating with OCR service.",
                )
            except Exception as e:
                logger.exception(f"Unexpected error during OCR processing: {e}")
                raise HTTPException(
                    status_code=500, detail=f"OCR processing failed: {e}"
                )

        return await retry_with_backoff(ocr_request)

    def build_ocr_messages(self, image_batch: List[Tuple[int, str]]) -> List[dict]:
        """
        Build the message payload for the OCR request.

        Args:
            image_batch (List[Tuple[int, str]]): List of tuples containing page numbers and image URLs.

        Returns:
            List[dict]: The message payload.
        """
        messages = [
            {
                "role": "system",
                "content": "You are an OCR assistant. Extract all text from the provided images (Describe images as if you're explaining them to a blind person eg: `[Image: In this picture, 8 people are posed hugging each other]`), which are attached to the document. Use markdown formatting for:\n\n- Headings (# for main, ## for sub)\n- Lists (- for unordered, 1. for ordered)\n- Emphasis (* for italics, ** for bold)\n- Links ([text](URL))\n- Tables (use markdown table format)\n\nFor non-text elements, describe them: [Image: Brief description]\n\nMaintain logical flow and use horizontal rules (---) to separate sections if needed. Adjust formatting to preserve readability.\n\nNote any issues or ambiguities at the end of your output.\n\nBe thorough and accurate in transcribing all text content.",
            },
            {
                "role": "user",
                "content": "Never skip any context! Convert document as is be creative to use markdown effectively to reproduce the same document by using markdown. Translate image text to markdown sequentially. Preserve order and completeness. Separate images with `---`. No skips or comments. Start with first image immediately.",
            },
        ]

        if len(image_batch) == 1:
            # Batch size = 1: Mention the specific page number
            page_num, img_url = image_batch[0]
            messages.append({
                "role": "user",
                "content": f"Page {page_num}:",
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}}
                ],
            })
        else:
            # Batch size >1: Include all page numbers and stress returning page numbers in response
            messages.append({
                "role": "user",
                "content": "Please perform OCR on the following images. Ensure that the extracted text includes the corresponding page numbers.",
            })
            content = []
            for page_num, img_url in image_batch:
                content.append({"type": "text", "text": f"Page {page_num}:"})
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            messages.append({
                "role": "user",
                "content": content,
            })

        return messages

    def extract_text_from_response(self, response) -> str:
        """
        Extract text from the OpenAI API response.

        Args:
            response: The response object from OpenAI API.

        Returns:
            str: The extracted text.

        Raises:
            HTTPException: If no text is extracted.
        """
        if (
            not response.choices
            or not hasattr(response.choices[0].message, "content")
            or not response.choices[0].message.content
        ):
            logger.warning("No text extracted from OCR.")
            raise HTTPException(
                status_code=500, detail="No text extracted from OCR."
            )

        extracted_text = response.choices[0].message.content.strip()
        logger.info(f"Extracted text length: {len(extracted_text)} characters.")
        return extracted_text

# Initialize OCR Service
ocr_service = OCRService()

# ----------------------------
# API Endpoints
# ----------------------------

@app.get("/ocr/methods")
async def list_ocr_methods():
    """List available OCR methods."""
    return {
        "methods": {
            "gpt4v": "Default GPT-4 Vision OCR",
            "tesseract": "Tesseract OCR",
            "hybrid": "Hybrid approach combining multiple OCR methods",
            "llm_whisperer": "LLM Whisperer API with layout preservation"
        }
    }

# @app.post("/ocr/llm_whisperer", response_model=OCRResponse,
#     summary="Process PDF using LLM Whisperer",
#     description="Extract text from a PDF using LLM Whisperer API with layout preservation")
# async def ocr_llm_whisperer_endpoint(
    request: Request,
    api_key: str = Security(get_api_key),
    file: Optional[UploadFile] = File(None),
    ocr_request: Optional[OCRRequest] = None,
):
    """Process PDF using LLM Whisperer API."""
    try:
        client = LLMWhispererClientV2()
        
        content_type = request.headers.get("content-type", "").lower()
        if content_type == "application/pdf":
            pdf_bytes = await request.body()
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="Empty PDF data")
        else:
            pdf_bytes = await get_pdf_bytes(file, ocr_request)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            tmp_pdf_file.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf_file.name
            
            try:
                # Process with LLM Whisperer
                result = await client.process_document(tmp_pdf_path)
                return OCRResponse(text=result)
            finally:
                os.remove(tmp_pdf_path)
                
    except Exception as e:
        logger.exception(f"Error in LLM Whisperer OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/tesseract", response_model=OCRResponse,
    summary="Process PDF using Tesseract OCR",
    description="Extract text from a PDF using Tesseract OCR.")
async def ocr_tesseract_endpoint(
    request: Request,
    api_key: str = Security(get_api_key),
    file: Optional[UploadFile] = File(None),
    ocr_request: Optional[OCRRequest] = None,
):
    """Process PDF using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        content_type = request.headers.get("content-type", "").lower()
        logger.info(f"Request content type: {content_type}")

        if content_type == "application/pdf":
            # Handle direct binary PDF upload
            pdf_bytes = await request.body()
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="Empty PDF data")
            logger.info(f"Received binary PDF upload, size: {len(pdf_bytes)} bytes")
        else:
            # Handle form upload or URL
            logger.info("Processing form upload or URL")
            pdf_bytes = await get_pdf_bytes(file, ocr_request)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            tmp_pdf_file.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf_file.name
            logger.info(f"Saved PDF to temporary file {tmp_pdf_path} for Tesseract OCR")

        try:
            # Convert PDF to images
            loop = asyncio.get_event_loop()
            image_bytes_list = await loop.run_in_executor(None, convert_pdf_to_images_pymupdf, tmp_pdf_path)

            texts = []
            page_count = len(image_bytes_list)
            logger.info(f"Processing {page_count} pages with Tesseract OCR")

            for page_num, img_bytes in image_bytes_list:
                logger.info(f"Processing page {page_num} with Tesseract OCR")
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img)
                texts.append(f"Page {page_num}:\n{text}")

            final_text = "\n\n".join(texts)
            logger.info(f"Tesseract OCR completed with {len(final_text)} characters extracted")
            return OCRResponse(text=final_text)
        finally:
            os.remove(tmp_pdf_path)
            logger.info(f"Deleted temporary PDF file {tmp_pdf_path}")
    except Exception as e:
        logger.exception(f"Error in Tesseract OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Tesseract OCR processing failed: {str(e)}")

@app.post("/ocr/hybrid")
async def ocr_hybrid_endpoint(
    request: Request,
    api_key: str = Security(get_api_key),
    file: Optional[UploadFile] = File(None),
    ocr_request: Optional[OCRRequest] = None,
):
    """Process PDF using both GPT-4V and Tesseract, combining results."""
    try:
        # For direct binary uploads, we need to create a copy of the request body
        # since it can only be read once
        content_type = request.headers.get("content-type", "").lower()
        if content_type == "application/pdf":
            pdf_bytes = await request.body()
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="Empty PDF data")

            # Create a modified request for each OCR method
            async def get_modified_request():
                class ModifiedRequest:
                    def __init__(self, headers, pdf_bytes):
                        self.headers = headers
                        self._body = pdf_bytes

                    async def body(self):
                        return self._body

                return ModifiedRequest(request.headers, pdf_bytes)

            # Create two separate requests with the same body
            gpt_request = await get_modified_request()
            tesseract_request = await get_modified_request()

            # Get GPT-4V results
            gpt_response = await ocr_endpoint(gpt_request, api_key, file, ocr_request)
            gpt_text = gpt_response.text

            # Get Tesseract results
            tesseract_response = await ocr_tesseract_endpoint(tesseract_request, api_key, file, ocr_request)
        else:
            # Get GPT-4V results
            gpt_response = await ocr_endpoint(request, api_key, file, ocr_request)
            gpt_text = gpt_response.text

            # Get Tesseract results
            tesseract_response = await ocr_tesseract_endpoint(request, api_key, file, ocr_request)
        tesseract_text = tesseract_response.text

        # Combine results with metadata
        combined_text = f"""
# GPT-4V Results
{gpt_text}

# Tesseract Results
{tesseract_text}
"""
        return OCRResponse(text=combined_text)
    except Exception as e:
        logger.exception(f"Error in hybrid OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Request



@app.post("/ocr", response_model=OCRResponse, 
    summary="Process PDF using GPT-4 Vision",
    description="""
    Extract text from a PDF using OpenAI's GPT-4 Vision model.

    This method provides high-accuracy OCR with:
    - Markdown formatting
    - Table structure preservation
    - Image descriptions
    """)
async def ocr_endpoint(
    request: Request,
    api_key: str = Security(get_api_key),
    file: Optional[UploadFile] = File(None),
    ocr_request: Optional[OCRRequest] = None,
):
    """
    Perform OCR on a provided PDF file or a PDF from a URL.

    Args:
        request (Request): The FastAPI request object
        api_key (str): API key for authentication
        file (Optional[UploadFile]): The uploaded PDF file.
        ocr_request (Optional[OCRRequest]): The OCR request containing a PDF URL.

    Returns:
        OCRResponse: The response containing the extracted text.

    Raises:
        HTTPException: If input validation fails or processing errors occur.
    """
    try:
        content_type = request.headers.get("content-type", "").lower()
        logger.info(f"Request content type: {content_type}")

        if content_type == "application/pdf":
            # Handle direct binary PDF upload
            pdf_bytes = await request.body()
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="Empty PDF data")
            logger.info(f"Received binary PDF upload, size: {len(pdf_bytes)} bytes")
        else:
            # Handle form upload or URL
            logger.info("Processing form upload or URL")
            pdf_bytes = await get_pdf_bytes(file, ocr_request)

        # Save PDF bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            tmp_pdf_file.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf_file.name
            logger.info(f"Saved PDF to temporary file {tmp_pdf_path}")

        try:
            # Process the PDF file
            loop = asyncio.get_event_loop()
            image_bytes_list = await loop.run_in_executor(None, convert_pdf_to_images_pymupdf, tmp_pdf_path)
            image_data_urls = encode_images(image_bytes_list)
            batches = create_batches(image_data_urls, Settings.BATCH_SIZE)
            extracted_texts = await process_batches(batches)
            final_text = concatenate_texts(extracted_texts)

            if not final_text:
                raise HTTPException(status_code=500, detail="OCR completed but no text was extracted")

            return OCRResponse(text=final_text)
        finally:
            os.remove(tmp_pdf_path)
            logger.info(f"Deleted temporary PDF file {tmp_pdf_path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled exception in /ocr endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during OCR processing."
        )
    try:
        # Retrieve PDF bytes
        pdf_bytes = await get_pdf_bytes(file, ocr_request)

        # Save PDF bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            tmp_pdf_file.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf_file.name
            logger.info(f"Saved PDF to temporary file {tmp_pdf_path}.")

        try:
            # Convert PDF to images using PyMuPDF with multiprocessing
            loop = asyncio.get_event_loop()
            image_bytes_list = await loop.run_in_executor(
                None, convert_pdf_to_images_pymupdf, tmp_pdf_path
            )

            # Encode images to base64 data URLs along with page numbers
            image_data_urls = encode_images(image_bytes_list)

            # Create batches for OCR
            batches = create_batches(image_data_urls, Settings.BATCH_SIZE)

            # Process OCR batches in parallel
            extracted_texts = await process_batches(batches)

            # Concatenate extracted texts
            final_text = concatenate_texts(extracted_texts)

            if not final_text:
                logger.warning("OCR completed but no text was extracted.")
                raise HTTPException(
                    status_code=500, detail="OCR completed but no text was extracted."
                )

            return OCRResponse(text=final_text)

        finally:
            # Clean up temporary PDF file
            os.remove(tmp_pdf_path)
            logger.info(f"Deleted temporary PDF file {tmp_pdf_path}.")

    except HTTPException:
        raise
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail="Invalid request parameters.")
    except Exception as e:
        logger.exception(f"Unhandled exception in /ocr endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during OCR processing.",
        )

# ----------------------------
# Helper Functions for API Endpoint
# ----------------------------

async def get_pdf_bytes(
    file: Optional[UploadFile],
    ocr_request: Optional[OCRRequest],
) -> bytes:
    """
    Retrieve PDF bytes from an uploaded file or a URL.

    Args:
        file (Optional[UploadFile]): The uploaded PDF file.
        ocr_request (Optional[OCRRequest]): The OCR request containing a PDF URL.

    Returns:
        bytes: The PDF content.

    Raises:
        HTTPException: If retrieval fails or input is invalid.
    """
    if not file and not ocr_request:
        logger.warning("No PDF file or URL provided in the request.")
        raise HTTPException(status_code=400, detail="No PDF file or URL provided.")

    if file and ocr_request and ocr_request.url:
        logger.warning("Both file and URL provided in the request; only one is allowed.")
        raise HTTPException(
            status_code=400, detail="Provide either a file or a URL, not both."
        )

    if file:
        return await read_uploaded_file(file)
    else:
        return download_pdf(ocr_request.url)

async def read_uploaded_file(file: UploadFile) -> bytes:
    """
    Read bytes from an uploaded file.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        bytes: The file content.

    Raises:
        HTTPException: If the file is invalid or reading fails.
    """
    allowed_types = ["application/pdf", "image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        logger.warning(f"Uploaded file has incorrect content type: {file.content_type}")
        raise HTTPException(
            status_code=400, detail=f"File must be PDF or image (JPEG/PNG). Got: {file.content_type}"
        )
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            logger.warning("Uploaded PDF file is empty.")
            raise HTTPException(
                status_code=400, detail="Uploaded PDF file is empty."
            )
        logger.info(f"Read uploaded PDF file, size: {len(pdf_bytes)} bytes.")
        return pdf_bytes
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to read uploaded file: {e}"
        )

async def process_batches(batches: List[List[Tuple[int, str]]]) -> List[str]:
    """
    Process each batch of images for OCR in parallel.

    Args:
        batches (List[List[Tuple[int, str]]]): List of image batches with page numbers.

    Returns:
        List[str]: Extracted texts from each batch.
    """
    tasks = [
        asyncio.create_task(ocr_service.perform_ocr_on_batch(batch))
        for batch in batches
    ]
    extracted_texts = await asyncio.gather(*tasks, return_exceptions=False)
    return extracted_texts

def concatenate_texts(texts: List[str]) -> str:
    """
    Concatenate a list of texts with double newlines.

    Args:
        texts (List[str]): List of text snippets.

    Returns:
        str: The concatenated text.
    """
    final_text = "\n\n".join(texts)
    logger.info(f"Total extracted text length: {len(final_text)} characters.")
    return final_text

# ----------------------------
# Run the Application
# ----------------------------

# To run the application, use the following command:
# uvicorn your_filename:app --reload
# Replace `your_filename` with the name of this Python file without the `.py` extension.