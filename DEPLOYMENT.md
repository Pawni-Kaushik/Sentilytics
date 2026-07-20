# Deploying Sentilytics — Sentiment Analyzer

Two parts: get the code on GitHub, then deploy it on Streamlit
Community Cloud (free, and the natural fit for a Streamlit app).

---

## Part 1 — Push to GitHub

From inside the `Sentilytics/` folder:

```bash
git init
git add .
git commit -m "Initial commit: Sentilytics Sentiment Analyzer"
```

Then on github.com: create a **new empty repository** (don't
initialize it with a README/license — you already have files), copy
its URL, and:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

**Before you push, double check `.env` isn't tracked** (it holds your
real GNews API key and must never be public):

```bash
git status
```

`.env` should NOT appear in the list of files to be committed — the
`.gitignore` already excludes it. If it does show up, remove it from
staging with `git rm --cached .env` before committing.

---

## Part 2 — Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io**, sign in with your GitHub account.
2. Click **"New app"**, pick your repo, branch `main`, and set the
   main file path to `app.py`.
3. Before/after deploying, open **Advanced settings** and:
   - Set **Python version to 3.12** (or 3.11) from the dropdown.
     TensorFlow does not yet publish wheels for very new Python
     releases (e.g. 3.13/3.14), and Streamlit Cloud's default Python
     version has been ahead of what TensorFlow supports. A
     `runtime.txt` (already included, set to `python-3.12`) is a
     best-effort hint, but it has been unreliable on Community Cloud
     recently — **explicitly picking the version in Advanced settings
     is the dependable way**. This can only be set at initial deploy;
     changing it later requires deleting and redeploying the app.
   - Open **Secrets** and paste:
   ```toml
   GNEWS_API_KEY = "your-real-key-here"
   ```
   Streamlit exposes this both as `st.secrets["GNEWS_API_KEY"]` and
   as a real environment variable, so the app's existing
   `os.getenv("GNEWS_API_KEY")` code works with no changes.
4. Click **Deploy**. First build will take a few minutes (TensorFlow
   is a large install).

### Live Search now uses a real API, not scraping
Earlier versions of this project used Playwright (a headless browser)
to scrape Reddit posts directly. That approach was dropped: Reddit
blocks datacenter/cloud IP ranges (which is what Streamlit Community
Cloud, and basically every other cloud host, run on) at the network
level, so the scrape failed 100% of the time once deployed even
though it worked locally. Live Search now calls the GNews API
(`news_service.py`) instead — a normal authenticated HTTPS request,
no browser, no system-level Chromium dependencies, and no IP
blocking. This is also why `packages.txt` was removed; it was only
ever needed for Playwright's Chromium system libraries.

### Resource use
Streamlit Community Cloud's free tier guarantees **1 GB of RAM** per
app. This project loads TensorFlow + two Keras models (sentiment +
NER), which is the main memory cost. Live Search itself is now just
a lightweight API call and shouldn't add meaningful memory pressure
on top of that.

### Redeploying after future changes
Once deployed, Cloud auto-redeploys on every `git push` to `main` —
no need to repeat the "New app" steps. Only `requirements.txt` /
`packages.txt` changes trigger a full rebuild; everything else
updates in near real time.
