import aiohttp
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote, urljoin
from typing import List
from dotenv import load_dotenv

# Configuration
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY')
USAGE_FILE = "api_usage.json"

def parse_date(date_str: str) -> datetime:
    """Mejorado para manejar más formatos de fecha y eliminar información de zona horaria."""
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%b %d, %Y",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str[:10], fmt)
            return dt
        except Exception:
            continue
    
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return datetime(1970, 1, 1)

def manage_usage(api_name: str, count: int = 1) -> bool:
    """Gestión mejorada de límites de API con reseteo temporal preciso"""
    try:
        if not os.path.exists(USAGE_FILE):
            usage = {
                "google": {"count": 0, "last_reset": datetime.now().isoformat()},
                "brave": {"count": 0, "last_reset": datetime.now().isoformat()}
            }
            with open(USAGE_FILE, "w") as f:
                json.dump(usage, f)
        
        with open(USAGE_FILE, "r+") as f:
            usage = json.load(f)
            now = datetime.now()
            
            if api_name == "google":
                last_reset = datetime.fromisoformat(usage["google"]["last_reset"])
                if now.date() > last_reset.date():
                    usage["google"]["count"] = 0
                    usage["google"]["last_reset"] = now.isoformat()
            elif api_name == "brave":
                last_reset = datetime.fromisoformat(usage["brave"]["last_reset"])
                if (now - last_reset).days >= 30:
                    usage["brave"]["count"] = 0
                    usage["brave"]["last_reset"] = now.isoformat()
            
            limits = {"google": 100, "brave": 2000}
            if usage[api_name]["count"] + count > limits[api_name]:
                return False
            
            usage[api_name]["count"] += count
            f.seek(0)
            json.dump(usage, f, indent=2)
            f.truncate()
            
            return True
            
    except Exception as e:
        print(f"Error en gestión de uso: {e}")
        return False

async def google_search(query: str) -> List[str]:
    if not manage_usage("google"):
        return []
    
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f'{query} -site:facebook.com -site:youtube.com -site:instagram.com -site:twitter.com -site:tiktok.com',
        "num": 10,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.googleapis.com/customsearch/v1", params=params) as res:
                data = await res.json()
                return [item['link'] for item in data.get('items', [])]
    
    except Exception as e:
        print(f"Error en Google Search: {e}")
        return []

async def brave_search(query: str) -> List[str]:
    if not manage_usage("brave"):
        return []
    
    params = {
        "q": f'{query}',   
    }
    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=200)
            ) as res:
                if res.status != 200:
                    error_body = await res.text()
                    print(f"Error Brave API ({res.status}): {error_body}")
                    return []
                
                data = await res.json()
                results = []
                for result in data.get('web', {}).get('results', []):
                    url = result.get('url') or result.get('link')
                    date_str = result.get('page_age') or result.get('date')
                    if url and date_str:
                        try:
                            published_date = datetime.fromisoformat(date_str)
                        except Exception:
                            published_date = datetime.min
                        results.append((url, published_date))
                
                results = sorted(results, key=lambda x: x[1], reverse=True)
                urls = [url for url, _ in results]
                return urls
                
    except Exception as e:
        print(f"Error en Brave Search: {e}")
        return []

def extract_ddg_url(href: str) -> str:
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    for param in ['uddg', 'u']:
        if param in qs:
            return unquote(qs[param][0])
    return href

async def duckduckgo_fallback(query: str) -> List[str]:
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": f"{query} -site:facebook.com -site:youtube.com -site:instagram.com -site:twitter.com -site:tiktok.com", "kp": "1"}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'es-ES,es;q=0.9'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params, headers=headers, timeout=10) as res:
                html = await res.text()
                soup = BeautifulSoup(html, 'html.parser')
                links = soup.select('a.result__url')
                return [extract_ddg_url(urljoin(search_url, link['href'])) for link in links[:10]]
    except Exception as e:
        print(f"Error DuckDuckGo: {e}")
        return []

async def crawl_web(query: str) -> List[str]:
    """
    Realiza búsquedas usando primero Google (hasta 100/día),
    luego Brave (hasta 2000/mes), y finalmente DuckDuckGo como fallback
    """
    urls = []
    
    urls = await google_search(query)
    if not urls:
        urls = await brave_search(query)
    if not urls:
        urls = await duckduckgo_fallback(query)
    
    return urls 