# Local image Q&A

This proof of concept indexes the photos in `images/` in Chroma and provides a
local browser UI for asking questions about them. It uses the configured
EmbeddingGemma 2 model for image/text embeddings and Ollama's `gemma3:4b` model
to generate answers.

## Run

1. Make sure Ollama is running and pull the answer-generation model:

   ```sh
   ollama pull gemma3:4b
   ```

2. Add supported photos (`.jpg`, `.jpeg`, `.png`, or `.webp`) to `images/`.
3. Start the app from the project directory:

   ```sh
   uv run main.py
   ```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

Each answer includes the retrieved image filenames and their relevance scores
(0 to 1; higher means more relevant). Only images scoring above 0.6 are sent
to the answer model and shown in the results.

The first run also downloads the configured embedding model if it is not
already available. Photos are indexed at startup and stored in `chroma_db/`.
Restart the app after adding or changing photos.