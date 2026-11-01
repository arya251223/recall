# API rate limits
The Helix REST API allows 600 requests per minute for each API key, with bursts of up to 100 requests per second. Requests over the limit receive an HTTP 429 response with a Retry-After header. Files larger than 64 MB must be uploaded using multipart upload. The maximum size of a single file is 5 GB.
