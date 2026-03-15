import streamlit as st
from utils.summarize import summarize
from utils.scraper import scrape_article
from utils.rouge_score import score as rouge_score, interpret
from utils.keywords import extract_keywords, extract_named_entities, get_reading_time

st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 News Summarizer")
    st.caption("Fine-tuned BART-large on CNN/DailyMail")
    st.divider()

    st.markdown("**Generation settings**")
    max_len   = st.slider("Max summary length (tokens)", 50, 256, 130)
    min_len   = st.slider("Min summary length (tokens)", 10, 80, 30)
    num_beams = st.select_slider("Beam search width", [1, 2, 4, 6], value=4)

    st.divider()
    st.markdown("**Model**")
    st.caption("Fine-tuned: `models/bart-finetuned`")
    st.caption("Fallback: `facebook/bart-large-cnn`")
    st.divider()
    st.markdown("**ROUGE-2 benchmarks**")
    st.caption("≥ 20 = Excellent")
    st.caption("14–20 = Good")
    st.caption("8–14 = Fair")
    st.caption("< 8 = Poor")


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Paste text", "From URL", "ROUGE evaluator"])

# ─────────────────────────────────────────────────────────────
# Tab 1 — Paste article text
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Paste an article")
    article = st.text_area("Article text", height=260,
                           placeholder="Paste any news article here...")
    ref_summary = st.text_area("Reference summary (optional — for ROUGE scoring)",
                               height=80,
                               placeholder="Paste the original summary to compare...")

    if st.button("Summarize", key="btn1"):
        if not article or len(article.split()) < 30:
            st.warning("Please paste a longer article (at least 30 words).")
        else:
            with st.spinner("Generating summary..."):
                result   = summarize(article, max_len=max_len,
                                     min_len=min_len, num_beams=num_beams)
                keywords = extract_keywords(article)
                entities = extract_named_entities(article)
                read_min = get_reading_time(article)

            st.divider()

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compression", f"{result['compression']}%")
            c2.metric("Input tokens",  result["input_tokens"])
            c3.metric("Output tokens", result["output_tokens"])
            c4.metric("Reading time",  f"{read_min} min")

            # Summary
            st.divider()
            st.markdown("**Summary**")
            st.success(result["summary"])

            # ROUGE if reference provided
            if ref_summary.strip():
                scores = rouge_score(result["summary"], ref_summary)
                verdict = interpret(scores["rouge2"])
                st.divider()
                st.markdown("**ROUGE scores vs your reference**")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("ROUGE-1", f"{scores['rouge1']:.1f}")
                r2.metric("ROUGE-2", f"{scores['rouge2']:.1f}")
                r3.metric("ROUGE-L", f"{scores['rougeL']:.1f}")
                r4.metric("Quality", verdict)

            # Keywords & entities
            st.divider()
            col_k, col_e = st.columns(2)
            with col_k:
                st.markdown("**Keywords**")
                st.write("  ".join(
                    f"`{k}`" for k in keywords
                ))
            with col_e:
                st.markdown("**Named entities**")
                if entities["people"]:
                    st.caption("People: " + ", ".join(entities["people"]))
                if entities["organizations"]:
                    st.caption("Orgs: " + ", ".join(entities["organizations"]))
                if entities["locations"]:
                    st.caption("Places: " + ", ".join(entities["locations"]))

# ─────────────────────────────────────────────────────────────
# Tab 2 — From URL
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Summarize from a URL")
    url = st.text_input("Article URL",
                        placeholder="https://www.thehindu.com/...")

    if st.button("Fetch & Summarize", key="btn2"):
        if not url.startswith("http"):
            st.warning("Enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Fetching article..."):
                scraped = scrape_article(url)

            if scraped["error"]:
                st.error(scraped["error"])
            else:
                st.info(f"**{scraped['title']}**  |  {len(scraped['text'].split())} words")

                with st.spinner("Generating summary..."):
                    result   = summarize(scraped["text"], max_len=max_len,
                                         min_len=min_len, num_beams=num_beams)
                    keywords = extract_keywords(scraped["text"])
                    entities = extract_named_entities(scraped["text"])
                    read_min = get_reading_time(scraped["text"])

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Compression", f"{result['compression']}%")
                c2.metric("Original words", len(scraped["text"].split()))
                c3.metric("Reading time", f"{read_min} min")

                st.divider()
                st.markdown("**Summary**")
                st.success(result["summary"])

                st.divider()
                col_k, col_e = st.columns(2)
                with col_k:
                    st.markdown("**Keywords**")
                    st.write("  ".join(f"`{k}`" for k in keywords))
                with col_e:
                    st.markdown("**Named entities**")
                    if entities["people"]:
                        st.caption("People: " + ", ".join(entities["people"]))
                    if entities["organizations"]:
                        st.caption("Orgs: " + ", ".join(entities["organizations"]))
                    if entities["locations"]:
                        st.caption("Places: " + ", ".join(entities["locations"]))

                with st.expander("View full article text"):
                    st.write(scraped["text"])

# ─────────────────────────────────────────────────────────────
# Tab 3 — ROUGE evaluator
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ROUGE score calculator")
    st.caption("Compare any two texts — model prediction vs reference summary.")

    col_a, col_b = st.columns(2)
    with col_a:
        pred = st.text_area("Model prediction / generated summary",
                            height=180)
    with col_b:
        ref = st.text_area("Reference summary (ground truth)",
                           height=180)

    if st.button("Compute ROUGE", key="btn3"):
        if not pred.strip() or not ref.strip():
            st.warning("Fill both fields.")
        else:
            scores  = rouge_score(pred, ref)
            verdict = interpret(scores["rouge2"])

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROUGE-1", f"{scores['rouge1']:.2f}")
            c2.metric("ROUGE-2", f"{scores['rouge2']:.2f}")
            c3.metric("ROUGE-L", f"{scores['rougeL']:.2f}")
            c4.metric("Quality", verdict)

            st.divider()
            st.markdown("**What these scores mean**")
            st.markdown("""
- **ROUGE-1** — overlap of individual words between prediction and reference
- **ROUGE-2** — overlap of word bigrams (pairs) — most commonly used benchmark
- **ROUGE-L** — longest common subsequence — rewards fluent, ordered overlap

*BART-large fine-tuned on CNN/DailyMail typically achieves ROUGE-2 ≈ 21–22.*
            """)
