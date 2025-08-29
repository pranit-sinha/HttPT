### Image Classification Usage Example

curl -X 'POST' \
  'http://localhost:8000/inference/image-classification' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "'$(base64 -i filename.ext | tr -d '\n')'",
    "datatype": "image"
  }'
