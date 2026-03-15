import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_article(url: str) -> dict:
    """
    Fetch and extract the main article text from a URL.
    Returns dict with keys: title, text, url, error.
    """
    try:
        # Try newspaper3k first (best for news sites)
        try:
            from newspaper import Article
            art = Article(url)
            art.download()
            art.parse()
            if art.text and len(art.text) > 200:
                return {
                    "title": art.title or "",
                    "text":  art.text,
                    "url":   url,
                    "error": None,
                }
        except Exception:
            pass

        # Fallback: BeautifulSoup
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav, footer, scripts
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form"]):
            tag.decompose()

        title = soup.find("h1")
        title = title.get_text(strip=True) if title else ""

        # Try article tag first, then main, then body
        for selector in ["article", "main", "[role='main']", "body"]:
            container = soup.select_one(selector)
            if container:
                paras = container.find_all("p")
                text  = "\n".join(p.get_text(strip=True) for p in paras
                                  if len(p.get_text(strip=True)) > 40)
                if len(text) > 200:
                    return {"title": title, "text": text,
                            "url": url, "error": None}

        return {"title": "", "text": "", "url": url,
                "error": "Could not extract article text from this URL."}

    except requests.exceptions.Timeout:
        return {"title": "", "text": "", "url": url,
                "error": "Request timed out. Try again."}
    except requests.exceptions.HTTPError as e:
        return {"title": "", "text": "", "url": url,
                "error": f"HTTP error: {e}"}
    except Exception as e:
        return {"title": "", "text": "", "url": url,
                "error": f"Unexpected error: {e}"}
