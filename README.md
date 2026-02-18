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

### The Gerri Bot

I built this not only for fellow Roman Roy heads but around a problem pretty common in automating retrieval for document stores with lots of semantic loading (e.g. "KYC", "Know Your Customer", and "customer identification procedures" all meaning the same thing). Keyword search only finds exact matches but embedding search can miss acronyms altogether. The natural thing to do is to run them together (my implementation uses BM25 against a custom analyzer that expands synonyms, and approximate k-nearest-neighbor search against dense sentence embeddings), then merge the two ranked lists. BM25 and cosine similarity scores can't be compared directly so you need a ranking method for the merge - I used reciprocal rank fusion. At this point the pipeline tended to flatten distinctions in similar units of text ("data retention requirements" vs. "data deletion rights"). This was resolved with a cross-encoder to re-score the top candidates by looking at queries and passages jointly.

The auditing pipeline itself is a four-node LangGraph state machine — retrieve, analyze, audit, report — and each node can fail independently without corrupting downstream state; the graph short-circuits to END if retrieval returns nothing so there's no risk of hallucinations. I was satisfied by how the gateway was able to handle this use case though I hadn't thought of anything interfacing with LangGraph during the design stage. I think it's especially neat that the endpoint could yield events upon each node completion for the caller's convenience though it sees the agent as a single entity. 
