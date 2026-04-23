# GeoAurora Backend 🌍✨

A high-performance FastAPI back-end service that powers the GeoAurora application. It acts as a Backend-For-Frontend (BFF), aggregating real-time Earth and Space intelligence from various scientific APIs, enriching the data using Generative AI, and delivering it with optimized latency.

## 🚀 Overview

The GeoAurora Backend is designed to consume complex, sometimes untidy scientific data streams and normalize them into consumer-friendly, unified JSON formats. It leverages smart caching strategies and asynchronous background tasks to ensure the frontend experiences zero latency when querying heavy datasets.

## 🌟 Key Features

### 1. Data Aggregation (NASA Integrations)
- **🌍 Earth Events (EONET):** Fetches and normalizes data on natural Earth events (wildfires, earthquakes, severe storms) from the last 30 days.
- **☀️ Space Weather (DONKI):** Aggregates multiple separate space weather feeds (Coronal Mass Ejections, Solar Flares, High-speed Solar Wind, and Solar Energetic Particles) into a single, unified chronological feed.
- **🪨 Near-Earth Objects (NEO):** Tracks asteroids approaching Earth in the upcoming week, calculating miss distances, velocity, and actively flagging potentially hazardous objects.

### 2. AI-Powered Enrichment (Gemini AI)
- **Auto-Enrichment:** Automatically cleans up missing or poorly formatted event descriptions using AI to generate professional, natural-language titles and summaries.
- **Detail Extraction:** Categorizes complex scientific events and extracts simple, plain-English explanations, severities (Low/Moderate/High/Extreme), and sources on-the-fly.
- **Space Facts:** Utilizes background threads to proactively generate unique, context-aware "fun facts" for the latest space weather events.

### 3. Performance & System Architecture
- **Distributed Caching:** Utilizes Redis Cloud with robust connection pooling to cache API responses and expensive AI generations (15-minute TTL for events, 7-day TTL for AI summaries).
- **Background Prefetching:** Dedicated async lifecycle workers (`refresher_loop`) proactively fetch and enrich data from NASA APIs every 15 minutes, preventing cache misses and long initial load times for users.
- **Concurrency Control:** Employs lightweight Redis-distributed locks to prevent "cache stampedes," ensuring that heavy AI-generation tasks are only processed once, even under heavy concurrent loads.
- **Rate-Limit Resiliency:** Built-in fault tolerance and graceful degradation mechanisms to handle secondary API rate limits.

---

## 🛠️ Setup & Local Development

### 1. Prerequisites
- Python 3.10+
- Redis Server (or a Redis Cloud instance)
- API Keys for NASA and Google Gemini

### 2. Installation
```bash
# Clone the repository
# Navigate to the project directory

# Create and activate a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add the following environment variables. **Never commit your `.env` file.**

```env
REDIS_URL=your_redis_connection_string
NASA_API_KEY=your_nasa_api_key_or_DEMO_KEY
GEMINI_API_KEY=your_google_gemini_api_key
```

### 4. Running the Server
```bash
uvicorn main:app --reload
```
The server will be available at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to view the auto-generated Swagger UI documentation for all endpoints.

---

## 📡 Core API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/eonet` | Returns tracked natural Earth events from the last 30 days. |
| `GET` | `/api/donki` | Returns aggregated and sorted space weather events. |
| `GET` | `/api/neo` | Returns near-Earth asteroids for the next 7 days, sorted by proximity. |
| `POST`| `/api/detail_enrich` | Enriches a raw event with AI categories, explanations, and severity. |

---

## ☁️ Deployment

This backend is designed to be easily deployed to modern cloud platforms like Render.com. 

1. Ensure all environment variables (`REDIS_URL`, `NASA_API_KEY`, `GEMINI_API_KEY`) are set securely in your hosting provider's dashboard.
2. A `render.yaml` configuration is included in the repository for streamlined CD (Continuous Deployment).
3. The server uses Uvicorn bound to host `0.0.0.0` and the platform-provided `$PORT`.

## 📜 License
MIT License
