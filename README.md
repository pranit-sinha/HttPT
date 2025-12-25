### Image Classification Usage Example

curl -X 'POST' \
  'http://localhost:8000/inference/image-classification' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "'$(base64 -i filename.ext | tr -d '\n')'",
    "datatype": "image"
  }'

### Chat Usage Example

curl -X 'POST' "http://localhost:8000/responses/stream"
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {"role": "user", "content": "Make a case for short men."}
  ],
  "model": "gemini-2.5-flash",
  "temperature": 0.7,
  "top_p": 0.6,
  "max_tokens": 1000, 
  "stream": true
}'
