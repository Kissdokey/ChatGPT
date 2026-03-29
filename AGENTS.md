# AGENTS.md

## Cursor Cloud specific instructions

This is a monorepo with two independent projects:

### 1. Feishu Chat Demo (`feishu-chat-demo/`)
- **Stack**: React 19 + TypeScript + Vite 8
- **Run dev**: `cd feishu-chat-demo && npm run dev`
- **Lint**: `cd feishu-chat-demo && npm run lint`
- **Build**: `cd feishu-chat-demo && npm run build`
- Fully self-contained with mock data; no external services needed.
- Default dev server port: `5173`.

### 2. Document AI (root `/`)
- **Stack**: Python 3 / Flask + LangChain + ChromaDB
- **Run**: `python server.py` (port 3001)
- **Requires**: A valid OpenAI API key. The code uses a custom proxy (`ai-proxy.ksord.com`) and a hardcoded key that is likely expired/invalid.
- `requirements.txt` is incomplete — `server.py` also imports `langchain`, `chromadb`, `pdfplumber`, and `python-docx`. Install all with: `pip install -r requirements.txt langchain chromadb pdfplumber python-docx`.
- `server.py` has hardcoded Windows paths for PDF/DOCX source files (lines 18–19, 30, 67). These will fail on Linux. The converted `.txt` files already exist in `data/` and `cut-data/cut/`.
- Running `server.py` on Linux will fail at startup due to the hardcoded Windows paths for `pdfplumber.open()` and `docx.Document()` calls at module level.

### Notes
- The two projects share no code, APIs, or data — they can be developed independently.
- For the Feishu Chat Demo, `npm install` is idempotent and sufficient for dependency refresh.
