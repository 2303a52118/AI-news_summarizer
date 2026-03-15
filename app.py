from flask import Flask, request, jsonify
from utils.summarize import summarize
from utils.scraper import scrape_article
from utils.rouge_score import score as rouge_score
from utils.keywords import extract_keywords, extract_named_entities, get_reading_time

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/summarize", methods=["POST"])
def summarize_text():
    """
    POST /summarize
    Body: { "text": "...", "max_len": 130, "min_len": 30 }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Provide 'text' in request body"}), 400

    text    = data["text"].strip()
    max_len = int(data.get("max_len", 130))
    min_len = int(data.get("min_len", 30))

    if len(text.split()) < 30:
        return jsonify({"error": "Text too short. Provide at least 30 words."}), 400

    result   = summarize(text, max_len=max_len, min_len=min_len)
    keywords = extract_keywords(text)
    entities = extract_named_entities(text)
    read_min = get_reading_time(text)

    return jsonify({
        **result,
        "keywords":    keywords,
        "entities":    entities,
        "reading_min": read_min,
    })


@app.route("/summarize-url", methods=["POST"])
def summarize_url():
    """
    POST /summarize-url
    Body: { "url": "https://..." }
    """
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Provide 'url' in request body"}), 400

    scraped = scrape_article(data["url"])
    if scraped["error"]:
        return jsonify({"error": scraped["error"]}), 400

    result   = summarize(scraped["text"])
    keywords = extract_keywords(scraped["text"])
    entities = extract_named_entities(scraped["text"])
    read_min = get_reading_time(scraped["text"])

    return jsonify({
        "title":    scraped["title"],
        "url":      scraped["url"],
        **result,
        "keywords": keywords,
        "entities": entities,
        "reading_min": read_min,
    })


@app.route("/rouge", methods=["POST"])
def compute_rouge():
    """
    POST /rouge
    Body: { "prediction": "...", "reference": "..." }
    """
    data = request.get_json()
    if not data or "prediction" not in data or "reference" not in data:
        return jsonify({"error": "Provide 'prediction' and 'reference'"}), 400

    scores = rouge_score(data["prediction"], data["reference"])
    return jsonify(scores)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
