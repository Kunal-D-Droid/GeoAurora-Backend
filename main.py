import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import redis
import json
import hashlib
import time
from datetime import datetime, timedelta, UTC
from contextlib import asynccontextmanager
import asyncio

load_dotenv()


NASA_API_KEY = os.getenv('NASA_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REDIS_URL = os.getenv('REDIS_URL')

# Validate Redis URL is configured
if not REDIS_URL:
    raise ValueError(
        "REDIS_URL environment variable is required. "
        "Please set it in your .env file with your Redis Cloud connection string."
    )

# Redis client with connection pooling and limits
# Configured to use Redis Cloud from .env file
redis_client = redis.Redis.from_url(
    REDIS_URL, 
    decode_responses=True, 
    socket_connect_timeout=5, 
    socket_timeout=5,
    max_connections=5,  # Limit max connections per instance
    retry_on_timeout=True,
    health_check_interval=30  # Check connection health
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Verify Redis Cloud connection
    try:
        redis_client.ping()
        print(f"✅ Redis Cloud connected successfully!")
        # Show partial URL for verification (hide password)
        redis_url_display = REDIS_URL.split('@')[1] if '@' in REDIS_URL else REDIS_URL[:50]
        print(f"   Connected to: {redis_url_display}")
    except Exception as e:
        print(f"❌ Redis Cloud connection failed: {e}")
        print(f"   Please check your REDIS_URL in .env file")
        print(f"   Current REDIS_URL: {REDIS_URL[:50]}..." if REDIS_URL else "   REDIS_URL is not set")
        # Don't raise error - let app start but cache operations will fail gracefully
    
    # Startup - refresher_loop will be defined later, but Python resolves at runtime
    try:
        asyncio.create_task(refresher_loop())
    except Exception as e:
        print('Failed to start refresher loop:', e)
    yield
    # Shutdown (if needed)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

class EventData(BaseModel):
    title: str
    description: str = None
class DetailEnrichRequest(BaseModel):
    title: str
    description: str | None = None
    date: str | None = None
    coordinates: list | None = None  # [lon, lat]

@app.post("/api/detail_enrich")
def detail_enrich(req: DetailEnrichRequest):
    """Use Gemini to produce rich details: category, severity, explanation (easy language), and curated sources.
    Results cached in Redis for 7 days by title.
    """
    if not isinstance(req.title, str) or not req.title.strip():
        raise HTTPException(status_code=400, detail="title required")
    cache_key = f"detail_enrich:{req.title.strip().lower()}"
    cached = redis_client.get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key missing for enrichment")

    try:
        # Lightweight lock to prevent duplicate Gemini calls for same key
        lock_key = cache_key + ":lock"
        got_lock = False
        try:
            got_lock = redis_client.set(lock_key, "1", nx=True, ex=60)  # 60s lock
        except Exception:
            got_lock = True  # proceed without lock if Redis issue
        if not got_lock:
            # Another request is computing; wait briefly for cache to appear
            for _ in range(12):  # up to ~6s
                awaitable = False
                try:
                    ready = redis_client.get(cache_key)
                except Exception:
                    ready = None
                if ready:
                    return json.loads(ready)
                import time
                time.sleep(0.5)
            # proceed anyway

        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        context_bits = []
        if req.date:
            context_bits.append(f"Date/Time: {req.date}")
        if req.coordinates and isinstance(req.coordinates, list) and len(req.coordinates) == 2:
            context_bits.append(f"Coordinates: lat {req.coordinates[1]}, lon {req.coordinates[0]}")
        context = ("\n".join(context_bits)).strip()
        prompt = (
            "You are an expert Earth/space science communicator. Given the event details, respond with a JSON object "
            "with keys: category (short noun phrase), severity (one of: Low, Moderate, High, Extreme), "
            "explanation (3-6 easy sentences, avoid jargon), sources (array of 0-3 high-quality article URLs).\n\n"
            f"Event Title: {req.title}\n"
            f"Event Description: {req.description or ''}\n"
            f"{('Context: ' + context) if context else ''}\n"
            "Return ONLY JSON."
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        resp = requests.post(gemini_url, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [{}])[0]
            .get('text', None)
        )
        result = {"category": None, "severity": None, "explanation": None, "sources": []}
        if text:
            import re, json as jsonlib
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    parsed = jsonlib.loads(match.group(0))
                    result.update({
                        "category": parsed.get("category"),
                        "severity": parsed.get("severity"),
                        "explanation": parsed.get("explanation"),
                        "sources": [u for u in (parsed.get("sources") or []) if isinstance(u, str) and u.startswith("http")][:3]
                    })
                except Exception:
                    pass
        # If still empty, try a lighter classification-only pass
        if not result.get("category") or not result.get("severity"):
            try:
                prompt2 = (
                    "Classify the event into a short category and severity. "
                    "Return JSON only with keys 'category' and 'severity' (Low/Moderate/High/Extreme).\n\n"
                    f"Title: {req.title}\nDescription: {req.description or ''}"
                )
                payload2 = {"contents": [{"parts": [{"text": prompt2}]}]}
                r2 = requests.post(gemini_url, headers=headers, json=payload2, timeout=25)
                r2.raise_for_status()
                d2 = r2.json()
                text2 = (
                    d2.get('candidates', [{}])[0]
                    .get('content', {})
                    .get('parts', [{}])[0]
                    .get('text', None)
                )
                if text2:
                    import re, json as jsonlib
                    m2 = re.search(r"\{[\s\S]*\}", text2)
                    if m2:
                        try:
                            p2 = jsonlib.loads(m2.group(0))
                            result["category"] = result.get("category") or p2.get("category")
                            result["severity"] = result.get("severity") or p2.get("severity")
                        except Exception:
                            pass
            except Exception:
                pass
        # Normalize and fill gaps
        title_low = (req.title or '').lower()
        desc_low = (req.description or '').lower()
        def infer_category() -> str | None:
            if any(k in title_low or k in desc_low for k in ["wildfire", "fire"]):
                return "Wildfire"
            if any(k in title_low or k in desc_low for k in ["earthquake", "seismic"]):
                return "Earthquake"
            if any(k in title_low or k in desc_low for k in ["cyclone", "hurricane", "typhoon", "storm"]):
                return "Tropical Cyclone"
            if any(k in title_low or k in desc_low for k in ["volcano", "eruption"]):
                return "Volcanic Activity"
            if any(k in title_low or k in desc_low for k in ["cme", "coronal mass ejection"]):
                return "Coronal Mass Ejection (CME)"
            if any(k in title_low or k in desc_low for k in ["solar flare", "flare"]):
                return "Solar Flare"
            if any(k in title_low or k in desc_low for k in ["sep", "solar energetic particles"]):
                return "Solar Energetic Particles (SEP)"
            if any(k in title_low or k in desc_low for k in ["coronal hole", "high-speed", "hss"]):
                return "High-speed Solar Wind / Coronal Hole"
            return None

        def normalize_severity(v: str | None) -> str | None:
            if not isinstance(v, str):
                return None
            s = v.strip().lower()
            if s in ["low", "minor", "weak", "slight"]:
                return "Low"
            if s in ["moderate", "medium"]:
                return "Moderate"
            if s in ["high", "severe", "strong"]:
                return "High"
            if s in ["extreme", "major", "intense"]:
                return "Extreme"
            return None

        if not result.get("category"):
            result["category"] = infer_category()
        sev_norm = normalize_severity(result.get("severity"))
        if not sev_norm:
            # heuristic by keywords
            if any(k in desc_low for k in ["evacu", "damage", "major", "dangerous", "life-threatening"]):
                sev_norm = "High"
            elif any(k in desc_low for k in ["watch", "warning", "moderate", "notable"]):
                sev_norm = "Moderate"
            else:
                sev_norm = None
        result["severity"] = sev_norm

        # Only cache non-empty results; otherwise cache a short-lived placeholder
        ttl = 60*60*24*7 if any([result.get("category"), result.get("severity"), result.get("explanation")]) else 60*10
        redis_client.setex(cache_key, ttl, json.dumps(result))
        try:
            redis_client.delete(lock_key)
        except Exception:
            pass
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import Request

def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    import re
    text = value.replace("null Miles", "").replace("NULL", "null")
    for bad in ["null", " None "]:
        text = text.replace(bad, " ")
    # Remove location-only tails like "18 Miles E from City, ST" or standalone "from City"
    text = re.sub(r"\b\d+\s*Miles?\s*[NSEW]{0,2}\s*from\b[^\n,]*[,\s]*[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*from\b[^\n]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.split()).strip(" ,")


def _normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    t = " ".join(title.split())
    return t.strip()


@app.get("/api/eonet")
def get_eonet(request: Request):
    print("/api/eonet endpoint hit")
    print("Request headers:", dict(request.headers))
    # Calculate date one month ago
    today = datetime.now(UTC).date()
    one_month_ago = today - timedelta(days=30)
    start_date = one_month_ago.isoformat()
    cache_key = f"eonet_events_{start_date}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            print("Cache hit for eonet_events")
            data = json.loads(cached)
        else:
            url = f"https://eonet.gsfc.nasa.gov/api/v3/events?start={start_date}"
            print(f"Fetching from NASA EONET API: {url}")
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Enforce basic sanitation when initially fetched
            if 'events' in data:
                for e in data['events']:
                    if 'title' in e:
                        e['title'] = _normalize_title(e.get('title', ''))
                    if 'description' in e and isinstance(e['description'], str):
                        e['description'] = _clean_text(e['description'])
                # Drop events that only contain null-like location texts
                data['events'] = [e for e in data['events'] if _clean_text(e.get('description', '')) or e.get('title')]
            # Cache for 15 minutes to align with frontend polling
            data["cached_at"] = datetime.now(UTC).isoformat() + "Z"
            redis_client.setex(cache_key, 900, json.dumps(data))
        # Ensure cached_at present
        if 'cached_at' not in data:
            data['cached_at'] = datetime.now(UTC).isoformat() + 'Z'
            print("Cached new EONET data")
        # For events missing title or description, use Gemini to generate them
        if 'events' in data:
            events = data['events'][:30]
            enhanced_events = []
            enrichment_count = 0
            max_enrichments = 5  # Limit enrichments per request to avoid rate limits
            
            for idx, e in enumerate(events):
                needs_title = not (isinstance(e.get('title'), str) and e.get('title').strip())
                needs_desc = not (isinstance(e.get('description'), str) and e.get('description').strip())
                if (needs_title or needs_desc) and GEMINI_API_KEY and enrichment_count < max_enrichments:
                    try:
                        # Add delay between requests to avoid rate limiting
                        if idx > 0:
                            time.sleep(1.5)  # 1.5 second delay between requests
                        
                        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                        prompt = f"Given this NASA event data: {json.dumps(e)}, generate a professional event title and a 2-3 sentence description. Respond in JSON as: {{'title': ..., 'description': ...}}. If you need to infer details, do so as an expert science communicator."
                        payload = {
                            "contents": [
                                {"parts": [{"text": prompt}]}
                            ]
                        }
                        headers = {
                            "Content-Type": "application/json",
                            "X-goog-api-key": GEMINI_API_KEY
                        }
                        resp = requests.post(gemini_url, headers=headers, json=payload, timeout=30)
                        
                        # Handle rate limiting gracefully
                        if resp.status_code == 429:
                            print(f"Gemini rate limit reached, skipping enrichment for remaining events")
                            break  # Stop trying to enrich remaining events
                        
                        resp.raise_for_status()
                        gemini_data = resp.json()
                        text = (
                            gemini_data.get('candidates', [{}])[0]
                            .get('content', {})
                            .get('parts', [{}])[0]
                            .get('text', None)
                        )
                        if text:
                            # Try to parse the JSON from Gemini's response
                            import re
                            import ast
                            match = re.search(r'\{.*\}', text, re.DOTALL)
                            if match:
                                try:
                                    ai_json = ast.literal_eval(match.group(0))
                                    if needs_title and 'title' in ai_json:
                                        e['title'] = ai_json['title']
                                    if needs_desc and 'description' in ai_json:
                                        e['description'] = ai_json['description']
                                    enrichment_count += 1
                                except Exception as parse_err:
                                    print('Failed to parse Gemini JSON:', parse_err)
                    except requests.exceptions.HTTPError as http_err:
                        if http_err.response.status_code == 429:
                            print('Gemini rate limit exceeded, skipping remaining enrichments')
                            break
                        else:
                            print('Gemini enrichment failed:', http_err)
                    except Exception as ai_err:
                        print('Gemini enrichment failed:', ai_err)
                enhanced_events.append(e)
            # Sanitize titles/descriptions and require geometry
            cleaned = []
            for e in enhanced_events:
                if not e.get('geometry'):
                    continue
                e['title'] = _normalize_title(e.get('title', ''))
                if isinstance(e.get('description'), str):
                    e['description'] = _clean_text(e['description'])
                cleaned.append(e)
            data['events'] = cleaned
            print(f"Returning {len(data['events'])} events from last month with enhanced titles/descriptions if needed.")
        return data
    except Exception as e:
        print("Exception in /api/eonet:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/donki")
def get_donki():
    """
    Aggregate multiple NASA DONKI feeds (CME, FLR, HSS, SEP), normalize to a common
    shape with a 'category' field, and return only the most recent records.
    Cached for 15 minutes.
    """
    now_utc = datetime.now(UTC)
    start_window_days = 30  # Fetch 1 month of data
    recency_hours = 720     # Show events from last 30 days (30 * 24 hours)
    cache_ttl_seconds = 15 * 60

    start_date = (now_utc - timedelta(days=start_window_days)).strftime("%Y-%m-%d")
    end_date = now_utc.strftime("%Y-%m-%d")
    cache_key = f"donki_mix:{start_date}:{end_date}"

    try:
        cached = None
        try:
            cached = redis_client.get(cache_key)
        except Exception as cache_err:
            print("DONKI cache read failed:", str(cache_err))
        if cached:
            combined = json.loads(cached)
        else:
            api_key = (NASA_API_KEY or 'DEMO_KEY')
            base = "https://api.nasa.gov/DONKI"

            def fetch(path: str):
                url = f"{base}/{path}?startDate={start_date}&endDate={end_date}&api_key={api_key}"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return r.json() or []

            try:
                cme = fetch("CME")
            except Exception as e:
                print("DONKI CME fetch failed:", e)
                cme = []
            try:
                flr = fetch("FLR")
            except Exception as e:
                print("DONKI FLR fetch failed:", e)
                flr = []
            try:
                hss = fetch("HSS")
            except Exception as e:
                print("DONKI HSS fetch failed:", e)
                hss = []
            try:
                sep = fetch("SEP")
            except Exception as e:
                print("DONKI SEP fetch failed:", e)
                sep = []

            combined = []
            # Normalize CME
            for it in cme:
                combined.append({
                    "activityID": it.get("activityID") or it.get("note", "")[:32],
                    "category": "Coronal Mass Ejection (CME)",
                    "note": it.get("note", ""),
                    "startTime": it.get("startTime"),
                    "link": it.get("link")
                })
            # Normalize Solar Flares
            for it in flr:
                combined.append({
                    "activityID": it.get("flrID") or it.get("instruments", [{}])[0].get("displayName", "")[:32],
                    "category": "Solar Flare",
                    "note": f"Class {it.get('classType','')} flare from {it.get('sourceLocation','')}.",
                    "startTime": it.get("beginTime") or it.get("peakTime"),
                    "link": it.get("link")
                })
            # Normalize High-speed streams / Coronal holes (HSS)
            for it in hss:
                combined.append({
                    "activityID": it.get("hssID") or (it.get("link", "")[:32]),
                    "category": "High-speed Solar Wind / Coronal Hole",
                    "note": it.get("link", "High-speed stream reported."),
                    "startTime": it.get("startTime"),
                    "link": it.get("link")
                })
            # Normalize Solar Energetic Particles (SEP)
            for it in sep:
                combined.append({
                    "activityID": it.get("sepID") or (it.get("link", "")[:32]),
                    "category": "Solar Energetic Particles (SEP)",
                    "note": it.get("link", "Solar energetic particle event reported."),
                    "startTime": it.get("eventTime") or it.get("startTime"),
                    "link": it.get("link")
                })

            try:
                redis_client.setex(cache_key, cache_ttl_seconds, json.dumps(combined))
            except Exception as cache_err:
                print("DONKI cache write failed:", str(cache_err))

        # Filter recent and sort
        recent_cutoff = now_utc - timedelta(hours=recency_hours)
        def parse_dt(s: str):
            try:
                dt_str = (s or "").replace("Z", "+00:00")
                parsed = datetime.fromisoformat(dt_str)
                # Ensure timezone-aware (if naive, assume UTC)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed
            except Exception:
                return None

        filtered = []
        for it in combined:
            dt = parse_dt(it.get("startTime"))
            if dt and dt >= recent_cutoff:
                filtered.append(it)

        # De-dupe by activityID/startTime+note
        seen = set()
        deduped = []
        for it in filtered:
            key = it.get("activityID") or f"{it.get('startTime','')}|{it.get('note','')}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        deduped.sort(key=lambda x: x.get("startTime", ""), reverse=True)
        
        # Pre-generate facts for the most recent events (background task)
        try:
            import threading
            threading.Thread(target=pre_generate_facts, args=(deduped[:5],), daemon=True).start()
        except Exception as e:
            print("Background fact generation failed:", e)
        
        return deduped[:20]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/neo")
def get_neo():
    """
    Fetch Near-Earth Objects (asteroids) from NASA NEO API.
    Returns asteroids approaching Earth in the next 7 days.
    Cached for 15 minutes.
    """
    now_utc = datetime.now(UTC)
    start_date = now_utc.strftime("%Y-%m-%d")
    # NEO API allows max 7 days ahead
    end_date = (now_utc + timedelta(days=7)).strftime("%Y-%m-%d")
    cache_ttl_seconds = 15 * 60
    cache_key = f"neo_asteroids:{start_date}:{end_date}"

    try:
        cached = None
        try:
            cached = redis_client.get(cache_key)
        except Exception as cache_err:
            print("NEO cache read failed:", str(cache_err))
        
        if cached:
            asteroids = json.loads(cached)
        else:
            api_key = (NASA_API_KEY or 'DEMO_KEY')
            url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api_key}"
            
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"NEO API fetch failed: {e}")
                return []

            asteroids = []
            near_earth_objects = data.get('near_earth_objects', {})
            
            # Flatten and normalize asteroid data
            for date_str, asteroid_list in near_earth_objects.items():
                for asteroid in asteroid_list:
                    # Get closest approach data
                    close_approach = asteroid.get('close_approach_data', [{}])[0] if asteroid.get('close_approach_data') else {}
                    
                    # Extract diameter range
                    diameter_meters = asteroid.get('estimated_diameter', {}).get('meters', {})
                    diameter_min = diameter_meters.get('estimated_diameter_min', 0)
                    diameter_max = diameter_meters.get('estimated_diameter_max', 0)
                    
                    # Extract miss distance and velocity
                    miss_distance_km = float(close_approach.get('miss_distance', {}).get('kilometers', '0'))
                    velocity_km_s = float(close_approach.get('relative_velocity', {}).get('kilometers_per_second', '0'))
                    
                    # Format closest approach date/time
                    approach_date = close_approach.get('close_approach_date_full') or close_approach.get('close_approach_date', date_str)
                    
                    # Create normalized asteroid event
                    asteroid_id = str(asteroid.get('id', ''))
                    asteroid_name = asteroid.get('name', 'Unknown Asteroid')
                    is_hazardous = asteroid.get('is_potentially_hazardous_asteroid', False)
                    
                    # Create description
                    size_desc = f"{diameter_min:.0f}-{diameter_max:.0f}m" if diameter_min > 0 else "Unknown size"
                    distance_desc = f"{miss_distance_km:,.0f} km" if miss_distance_km > 0 else "Unknown distance"
                    velocity_desc = f"{velocity_km_s:.2f} km/s" if velocity_km_s > 0 else "Unknown velocity"
                    
                    description = f"Asteroid {asteroid_name} approaching Earth. Size: {size_desc}, Distance: {distance_desc}, Velocity: {velocity_desc}."
                    if is_hazardous:
                        description += " ⚠️ Potentially hazardous asteroid."
                    
                    note = f"Size: {size_desc} | Distance: {distance_desc} | Velocity: {velocity_desc}"
                    if is_hazardous:
                        note += " | ⚠️ HAZARDOUS"
                    
                    asteroids.append({
                        "activityID": asteroid_id,
                        "category": "Near-Earth Asteroid",
                        "title": f"{asteroid_name} ({size_desc})",
                        "description": description,
                        "startTime": approach_date,
                        "note": note,
                        "link": asteroid.get('nasa_jpl_url', ''),
                        "hazardous": is_hazardous,
                        "diameter_min": diameter_min,
                        "diameter_max": diameter_max,
                        "velocity": velocity_km_s,
                        "miss_distance": miss_distance_km,
                        "orbiting_body": close_approach.get('orbiting_body', 'Earth')
                    })
            
            # Sort by miss_distance (closest first)
            asteroids.sort(key=lambda x: x.get('miss_distance', float('inf')))
            
            # Cache the results
            try:
                redis_client.setex(cache_key, cache_ttl_seconds, json.dumps(asteroids))
            except Exception as cache_err:
                print("NEO cache write failed:", str(cache_err))
        
        # Return top 20 closest asteroids
        return asteroids[:20]
    except Exception as e:
        print(f"NEO endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def pre_generate_facts(events):
    """Pre-generate facts for events in the background to improve user experience"""
    # Skip if API key is not configured
    if not GEMINI_API_KEY:
        return
    
    try:
        for event in events:
            stable_id = event.get("activityID") or f"{event.get('startTime','')}|{event.get('note','')}"
            cache_key = f"summary:{stable_id}"
            
            # Check if already cached
            try:
                if redis_client.exists(cache_key):
                    continue
            except Exception:
                pass
            
            # Generate summary with enhanced prompt
            title = event.get("note", "Space Weather Event")
            description = f"Event Type: {event.get('category', 'Space Weather Event')}\nEvent ID: {stable_id}\nTimestamp: {event.get('startTime', 'Recent')}"
            
            try:
                # Use Gemini API to generate summary (consistent with other endpoints)
                gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"""Summarize this NASA space weather event in 2-3 sentences in plain English. Then, add a unique, specific fun fact about this particular event or its type, starting with 'Fun Fact:' on a new line.

Requirements for the fun fact:
1. Make it specific to THIS particular event (not generic)
2. Include specific details like speed, class, location, or timing if available
3. Make it educational and interesting
4. Avoid generic statements like "solar flares are..." or "CMEs can travel..."
5. Focus on what makes THIS event unique or notable

Event: {title}
Description: {description}"""}]
                    }]
                }
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": GEMINI_API_KEY
                }
                response = requests.post(gemini_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if "candidates" in result and len(result["candidates"]) > 0:
                    summary = result["candidates"][0]["content"]["parts"][0]["text"]
                    # Cache the result
                    try:
                        redis_client.setex(cache_key, 60*60*24*7, summary)  # Cache for 7 days
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to pre-generate fact for {stable_id}: {e}")
    except Exception as e:
        print(f"Pre-generation task failed: {e}")


@app.get("/api/summary")
def summary_info():
    """Get information about the /api/summary endpoint."""
    return {
        "message": "This endpoint requires a POST request with JSON body",
        "method": "POST",
        "endpoint": "/api/summary",
        "required_fields": {
            "title": "string (required)",
            "description": "string (optional)"
        },
        "example_request": {
            "title": "Solar Flare X1.2",
            "description": "A strong solar flare detected on the Sun"
        },
        "example_response": {
            "summary": "Summary text with fun fact..."
        }
    }

@app.get("/")
def root():
    return {"message": "GeoAurora Python API", "endpoints": ["/api/eonet", "/api/donki", "/api/neo", "/api/summary"]}


# -----------------------------
# Background refresher (15 min)
# -----------------------------

async def refresh_eonet_cache() -> None:
    """Prefetch EONET, enrich missing fields, filter, and store in Redis (15 min TTL)."""
    today = datetime.now(UTC).date()
    one_month_ago = today - timedelta(days=30)
    start_date = one_month_ago.isoformat()
    cache_key = f"eonet_events_{start_date}"
    try:
        url = f"https://eonet.gsfc.nasa.gov/api/v3/events?start={start_date}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if 'events' in data:
            events = data['events'][:30]
            enhanced_events = []
            enrichment_count = 0
            max_enrichments = 3  # Limit enrichments in background refresh to avoid rate limits
            
            for idx, e in enumerate(events):
                needs_title = not (isinstance(e.get('title'), str) and e.get('title').strip())
                needs_desc = not (isinstance(e.get('description'), str) and e.get('description').strip())
                if (needs_title or needs_desc) and GEMINI_API_KEY and enrichment_count < max_enrichments:
                    try:
                        # Add longer delay for background tasks to avoid rate limiting
                        if idx > 0:
                            time.sleep(3)  # 3 second delay between requests in background
                        
                        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
                        prompt = f"Given this NASA event data: {json.dumps(e)}, generate a professional event title and a 2-3 sentence description. Respond in JSON as: {'{'}'title': ..., 'description': ...{'}'}. If you need to infer details, do so as an expert science communicator."
                        payload = {"contents": [{"parts": [{"text": prompt}]}]}
                        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
                        r = requests.post(gemini_url, headers=headers, json=payload, timeout=30)
                        
                        # Handle rate limiting gracefully
                        if r.status_code == 429:
                            print('Gemini rate limit in background refresh, skipping remaining')
                            break  # Stop trying to enrich remaining events
                        
                        r.raise_for_status()
                        gemini_data = r.json()
                        text = (
                            gemini_data.get('candidates', [{}])[0]
                            .get('content', {})
                            .get('parts', [{}])[0]
                            .get('text', None)
                        )
                        if text:
                            import re, ast
                            match = re.search(r'\{.*\}', text, re.DOTALL)
                            if match:
                                try:
                                    ai_json = ast.literal_eval(match.group(0))
                                    if needs_title and 'title' in ai_json:
                                        e['title'] = ai_json['title']
                                    if needs_desc and 'description' in ai_json:
                                        e['description'] = ai_json['description']
                                    enrichment_count += 1
                                except Exception:
                                    pass
                    except requests.exceptions.HTTPError as http_err:
                        if http_err.response.status_code == 429:
                            print('Gemini rate limit exceeded in background refresh, skipping')
                            break
                        else:
                            print('Gemini enrichment (prefetch) failed:', http_err)
                    except Exception as ai_err:
                        print('Gemini enrichment (prefetch) failed:', ai_err)
                enhanced_events.append(e)
            data['events'] = [e for e in enhanced_events if e.get('geometry')]
        # 15-min TTL
        redis_client.setex(cache_key, 900, json.dumps(data))
    except Exception as e:
        print('EONET prefetch failed:', e)


async def refresh_donki_cache() -> None:
    """Prefetch DONKI mixed categories and store in Redis (15 min TTL)."""
    now_utc = datetime.now(UTC)
    start_window_days = 30  # Fetch 1 month of data
    start_date = (now_utc - timedelta(days=start_window_days)).strftime("%Y-%m-%d")
    end_date = now_utc.strftime("%Y-%m-%d")
    cache_key = f"donki_mix:{start_date}:{end_date}"
    try:
        # Reuse the same aggregator as the endpoint
        _ = get_donki()  # will rebuild cache on miss
        # If we reached here, cache should be warm already due to setex in get_donki
        # But to be explicit, read and bump TTL
        cached = redis_client.get(cache_key)
        if cached:
            redis_client.setex(cache_key, 900, cached)
    except Exception as e:
        print('DONKI prefetch failed:', e)


async def refresh_neo_cache() -> None:
    """Prefetch NEO asteroids and store in Redis (15 min TTL)."""
    now_utc = datetime.now(UTC)
    start_date = now_utc.strftime("%Y-%m-%d")
    end_date = (now_utc + timedelta(days=7)).strftime("%Y-%m-%d")
    cache_key = f"neo_asteroids:{start_date}:{end_date}"
    try:
        # Trigger the endpoint to rebuild cache if needed
        _ = get_neo()  # This will rebuild cache on miss
        # Read and bump TTL
        cached = redis_client.get(cache_key)
        if cached:
            redis_client.setex(cache_key, 900, cached)
    except Exception as e:
        print('NEO prefetch failed:', e)


async def refresher_loop() -> None:
    # Initial delay a bit to allow server to come up
    await asyncio.sleep(2)
    while True:
        await refresh_eonet_cache()
        await refresh_donki_cache()
        await refresh_neo_cache()
        await asyncio.sleep(15 * 60)


# -----------------------------
# On-demand auto enrichment API
# -----------------------------

class AutoEnrichRequest(BaseModel):
    id: str | None = None
    title: str
    description: str | None = None

@app.post("/api/auto_enrich")
def auto_enrich(req: AutoEnrichRequest):
    """If title/description are missing or short, ask Gemini for a concise description.
    Caches by event id or title for 7 days.
    """
    if not req.title or not isinstance(req.title, str):
        raise HTTPException(status_code=400, detail="title required")
    key_base = (req.id or req.title).strip().lower()
    cache_key = f"auto_enrich:{key_base}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Decide if enrichment needed
    needs = False
    if not req.description or len(req.description.strip()) < 40:
        needs = True
    if not needs:
        return {"description": req.description}

    if not GEMINI_API_KEY:
        return {"description": req.description}

    try:
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        prompt = (
            "Write a 2-4 sentence, easy-to-read description of this NASA event. Avoid boilerplate.\n"
            f"Title: {req.title}\nExisting Description: {req.description or ''}\n"
            "Return only the text (no JSON)."
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        r = requests.post(gemini_url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        d = r.json()
        text = (
            d.get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [{}])[0]
            .get('text', '')
        )
        result = {"description": _clean_text(text) or (req.description or '')}
        redis_client.setex(cache_key, 60*60*24*7, json.dumps(result))
        return result
    except Exception as e:
        return {"description": req.description or ''}


class KeyTermsRequest(BaseModel):
    title: str
    description: str | None = None
    summary: str | None = None

@app.post("/api/key_terms")
def get_key_terms(req: KeyTermsRequest):
    """Extract and define key technical terms from event content using Gemini."""
    if not req.title or not isinstance(req.title, str):
        raise HTTPException(status_code=400, detail="title required")
    
    # Combine all text content
    full_text = f"{req.title} {req.description or ''} {req.summary or ''}".strip()
    
    cache_key = f"key_terms:{hashlib.md5(full_text.encode()).hexdigest()}"
    print(f"Key terms request for title: '{req.title[:50]}...' -> cache key: {cache_key}")
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            print(f"Key terms cache hit for key: {cache_key}")
            return json.loads(cached)
    except Exception as cache_err:
        print(f"Failed to read key terms from cache: {cache_err}")
        # Continue to generate new terms
    
    if not GEMINI_API_KEY:
        return {"terms": []}
    
    try:
        # Get existing key terms to avoid duplicates
        existing_terms = _get_existing_key_terms()
        
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        prompt = f"""Analyze this event content and identify 3-5 key technical or scientific terms that a general audience might not understand. For each term, provide a simple, clear definition.

Event Content:
Title: {req.title}
Description: {req.description or ''}
Summary: {req.summary or ''}

Requirements:
1. Focus on terms specific to THIS particular event
2. Avoid generic terms like "wildfire", "satellite", "earthquake" unless they have specific technical aspects
3. Prioritize unique or specialized terms related to this event's location, type, or characteristics
4. Each term should be genuinely technical/scientific and educational

Return the response in this exact JSON format:
{{
  "terms": [
    {{
      "term": "Term Name",
      "definition": "Simple, clear definition in 1-2 sentences"
    }}
  ]
}}

Existing terms to avoid repeating: {list(existing_terms.keys())[:10] if existing_terms else 'None'}

Focus on terms that are:
- Specific to this event's location or type
- Technical/scientific concepts not commonly known
- Unique to this particular occurrence"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        r = requests.post(gemini_url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        d = r.json()
        result_text = (
            d.get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [{}])[0]
            .get('text', '')
        )
        
        # Try to parse JSON response
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            elif "```" in result_text:
                json_start = result_text.find("```") + 3
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            
            result = json.loads(result_text)
            
            # Validate structure
            if not isinstance(result, dict) or "terms" not in result:
                result = {"terms": []}
            if not isinstance(result["terms"], list):
                result["terms"] = []
                
            # Clean up terms
            cleaned_terms = []
            for term in result["terms"]:
                if isinstance(term, dict) and "term" in term and "definition" in term:
                    cleaned_terms.append({
                        "term": term["term"].strip(),
                        "definition": term["definition"].strip()
                    })
            result["terms"] = cleaned_terms
            
        except json.JSONDecodeError:
            result = {"terms": []}
        
        try:
            redis_client.setex(cache_key, 7 * 24 * 3600, json.dumps(result))  # 7 days
            print(f"Key terms cached successfully for key: {cache_key}")
        except Exception as cache_err:
            print(f"Failed to cache key terms: {cache_err}")
            # Continue without caching rather than failing completely
            
        return result
        
    except Exception as e:
        return {"terms": []}


def _get_existing_facts() -> list:
    """Get all existing fun facts from Redis to avoid duplicates."""
    try:
        # Get all summary cache keys
        keys = redis_client.keys("detail_summary:*")
        facts = []
        for key in keys:
            try:
                cached = redis_client.get(key)
                if cached:
                    text = cached.decode('utf-8')
                    if 'Fun Fact:' in text:
                        fact = text.split('Fun Fact:')[1].strip()
                        if fact and len(fact) > 20:  # Only meaningful facts
                            facts.append(fact.lower())
            except:
                continue
        return facts
    except:
        return []


def _get_existing_key_terms() -> dict:
    """Get all existing key terms from Redis to avoid duplicates."""
    try:
        keys = redis_client.keys("key_terms:*")
        existing_terms = {}
        for key in keys:
            try:
                cached = redis_client.get(key)
                if cached:
                    data = json.loads(cached)
                    if isinstance(data, dict) and 'terms' in data:
                        for term in data['terms']:
                            if isinstance(term, dict) and 'term' in term:
                                term_name = term['term'].lower().strip()
                                existing_terms[term_name] = term.get('definition', '')
            except:
                continue
        return existing_terms
    except:
        return {}


@app.post("/api/summary")
def generate_summary(event: EventData):
    """Generate summary with deduplication for fun facts."""
    title = event.title if isinstance(event.title, str) else ""
    description = event.description if event.description else ""
    
    if not title or not title.strip():
        raise HTTPException(status_code=400, detail="Title required")
    
    cache_key = f"detail_summary:{title.lower().strip()}"
    
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return {"summary": cached.decode('utf-8')}
    except:
        pass
    
    if not GEMINI_API_KEY:
        return {"summary": f"{title}: {description}"}
    
    try:
        # Get existing facts to avoid duplicates
        existing_facts = _get_existing_facts()
        
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Enhanced prompt with deduplication instructions
        prompt = f"""Write a 2-3 sentence summary of this NASA space weather event, followed by a unique "Did You Know?" fact.

Event: {title}
Description: {description}

Requirements:
1. Summary: Explain what happened in simple terms
2. Fun Fact: Provide a unique, interesting fact about this specific event or its type
3. Make the fact specific to this event's characteristics, timing, or unique details
4. Include specific data like speed, class, location, or timing if available
5. Avoid generic statements like "solar flares are..." or "CMEs can travel..."
6. Focus on what makes THIS particular event notable or unique
7. Ensure the fact is educational and engaging

Format: [Summary text]\n\nFun Fact: [Unique fact about this specific event]

Existing facts to avoid repeating: {existing_facts[:5] if existing_facts else 'None'}"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        r = requests.post(gemini_url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        d = r.json()
        text = (
            d.get('candidates', [{}])[0]
            .get('content', {})
            .get('parts', [{}])[0]
            .get('text', '')
        )
        
        result = _clean_text(text) or f"{title}: {description}"
        
        try:
            redis_client.setex(cache_key, 7 * 24 * 3600, result)  # 7 days
        except:
            pass
            
        return {"summary": result}
        
    except Exception as e:
        return {"summary": f"{title}: {description}"}

# Catch-all route for unwanted requests (like browser extensions)
@app.post("/api/v2/scan/url")
def handle_scan_url():
    """Handle unwanted scan requests from browser extensions"""
    return {"message": "Endpoint not available", "status": "ignored"}

@app.get("/api/v2/scan/url")
def handle_scan_url_get():
    """Handle unwanted scan requests from browser extensions"""
    return {"message": "Endpoint not available", "status": "ignored"}

if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment variable (Render provides this) or default to 8000 for local
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"  # 0.0.0.0 for Render, 127.0.0.1 for local
    uvicorn.run(app, host=host, port=port)
