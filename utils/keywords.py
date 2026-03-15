import re
from collections import Counter


# Simple keyword extraction without spaCy dependency
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "will","would","could","should","may","might","that","this","these",
    "those","it","its","he","she","they","we","you","i","said","says",
    "also","than","then","when","where","which","who","how","what","as",
    "about","after","before","during","while","between","into","through",
    "not","no","so","if","up","out","new","more","other","one","two",
    "all","some","their","there","been","being","do","did","does"
}


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    words = [w for w in words if w not in STOPWORDS]
    freq  = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


def extract_named_entities(text: str) -> dict:
    """
    Lightweight NER using regex heuristics.
    Returns dict with people, organizations, locations.
    """
    # Capitalized multi-word phrases (likely names/orgs/places)
    pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'
    candidates = re.findall(pattern, text)
    freq = Counter(candidates)

    # Simple heuristics to separate types
    org_hints  = {"Inc","Corp","Ltd","Co","Group","Bank","Fund","Institute",
                  "University","College","Ministry","Department","Agency",
                  "Association","Foundation","Committee","Council","Party"}
    loc_hints  = {"Street","Avenue","Road","City","State","Country","River",
                  "Mountain","Lake","Ocean","Sea","Bay","Island","Province",
                  "District","Region","County","Town","Village"}

    people, orgs, locations = [], [], []
    for name, _ in freq.most_common(20):
        words = name.split()
        last  = words[-1]
        if last in org_hints or any(w in org_hints for w in words):
            orgs.append(name)
        elif last in loc_hints or any(w in loc_hints for w in words):
            locations.append(name)
        else:
            people.append(name)

    return {
        "people":        people[:5],
        "organizations": orgs[:5],
        "locations":     locations[:5],
    }


def get_reading_time(text: str) -> int:
    """Estimated reading time in minutes (avg 200 wpm)."""
    words = len(text.split())
    return max(1, round(words / 200))
