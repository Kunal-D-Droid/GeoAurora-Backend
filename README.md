# GeoAurora Backend

FastAPI backend for GeoAurora - Real-time Earth and Space events from NASA APIs.

## Features

- 🌍 **NASA EONET** - Earth natural events (wildfires, earthquakes, storms, etc.)
- ☀️ **NASA DONKI** - Space weather events (CME, solar flares, HSS, SEP)
- 🪨 **NASA NEO** - Near-Earth asteroids tracking
- 🤖 **AI-Powered Summaries** - Gemini API for intelligent event descriptions
- ⚡ **Redis Cloud Caching** - Fast response times with 15-minute cache for events, 7-day cache for summaries
- 🔄 **Background Refresh** - Automatic cache updates every 15 minutes

## Setup

### Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file** in the `Backend` folder:
   ```env
   REDIS_URL=your_redis_cloud_connection_string
   NASA_API_KEY=your_nasa_api_key
   GEMINI_API_KEY=your_google_gemini_api_key
   HF_API_KEY=your_huggingface_api_key (optional)
   ```

5. **Run the server:**
   ```bash
   python main.py
   ```
   Or with auto-reload:
   ```bash
   uvicorn main:app --reload
   ```

   Server will start at `http://127.0.0.1:8000`

## API Endpoints

- `GET /` - API information
- `GET /api/eonet` - Earth events from NASA EONET
- `GET /api/donki` - Space weather events from NASA DONKI
- `GET /api/neo` - Near-Earth asteroids from NASA NEO
- `POST /api/summary` - Generate AI summary with fun facts
- `POST /api/detail_enrich` - Enrich event details
- `POST /api/key_terms` - Extract key terms
- `POST /api/auto_enrich` - Auto-enrich missing fields

## Deployment

### Render.com

1. Push code to GitHub
2. Connect repository to Render
3. Create new Web Service
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard
6. Deploy!

The `render.yaml` file is included for easy deployment configuration.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_URL` | ✅ Yes | Redis Cloud connection string |
| `NASA_API_KEY` | ✅ Yes | NASA API key from [api.nasa.gov](https://api.nasa.gov) |
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `HF_API_KEY` | ⚠️ Optional | Hugging Face API key (fallback for summaries) |

## Caching Strategy

- **Event Data (EONET, DONKI, NEO):** 15 minutes TTL
- **AI Summaries:** 7 days TTL
- **Background Refresh:** Automatic every 15 minutes

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Redis** - Caching layer (Redis Cloud)
- **Google Gemini API** - AI-powered content generation
- **NASA APIs** - EONET, DONKI, NEO

## License

MIT

