#!/usr/bin/env python3
"""
cbse_papers_crawler.py
Simple crawler to download CBSE question papers and store them as:
 output/{year}/{class}/{subject}/{file.pdf}
"""

import time
import random
import re
import os
import io
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse
import logging

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib.robotparser

# ---------- CONFIG ----------
START_URL = "https://www.cbse.gov.in/cbsenew/question-paper.html"
OUTPUT_DIR = Path("output")
USER_AGENTS = [
    # a few common UA strings (rotate)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
    " Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/120.0.0.0 Safari/537.36",
]
MIN_DELAY = 3      # seconds
MAX_DELAY = 12     # seconds
MAX_RETRIES = 3
TIMEOUT = 30
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def polite_sleep():
    # random sleep to mimic human browsing
    t = random.uniform(MIN_DELAY, MAX_DELAY)
    logging.debug(f"sleeping {t:.1f}s")
    time.sleep(t)


def allowed_by_robots(target_url, user_agent="*"):
    rp = urllib.robotparser.RobotFileParser()
    parsed = urlparse(target_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, target_url)
    except Exception as e:
        logging.warning(f"robots.txt could not be fetched/parsed ({e}); proceeding cautiously")
        # conservative: allow if robots failed to load? Here return True but keep polite rate limits.
        return True


def get_session():
    s = requests.Session()
    s.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        # rotate UA later per request
    })
    return s


def download_binary(session, url, referer=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {}
            headers["User-Agent"] = random.choice(USER_AGENTS)
            if referer:
                headers["Referer"] = referer
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                content = r.content  # small-to-medium files; adjust for huge single files
                return content, r.headers.get("Content-Type", "")
        except Exception as e:
            logging.warning(f"download attempt {attempt} failed for {url}: {e}")
            time.sleep(2 * attempt)
    logging.error(f"giving up downloading {url}")
    return None, None


def save_pdf_bytes(target_path: Path, data: bytes):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(data)


def extract_pdfs_from_zip_bytes(zip_bytes: bytes, target_folder: Path):
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        logging.error("Bad zip file - skipping")
        return []
    saved = []
    for zi in z.infolist():
        if zi.filename.lower().endswith(".pdf"):
            fname = Path(zi.filename).name
            out = target_folder / fname
            # avoid path traversal inside zip
            if ".." in Path(zi.filename).parts:
                logging.warning(f"Skipping suspicious zip entry {zi.filename}")
                continue
            with z.open(zi) as src, open(out, "wb") as dst:
                dst.write(src.read())
            saved.append(out)
    return saved


def make_filename_from_subject(subject_text, href):
    # sanitize
    name = re.sub(r"[^\w\-_\. ]", "_", subject_text.strip())
    # keep extension from href if present
    ext = Path(urlparse(href).path).suffix or ".pdf"
    return name + ext


def parse_and_queue_downloads(page_html, base_url):
    soup = BeautifulSoup(page_html, "html.parser")
    downloads = []  # list of dict {year, cls, subject, url}

    # Strategy:
    # - Walk through headings; remember current year and class when we see them
    # - For each row/line that represents a subject, try to find a link (.pdf/.zip) or a JS onclick data with URL.

    current_year = "unknown_year"
    current_class = "unknown_class"

    # Heuristics: find elements that look like "Question Papers for Examination 2025"
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "p", "div"]):
        txt = (tag.get_text(separator=" ", strip=True) or "").upper()
        # detect year
        m = re.search(r"(EXAMINATION|QUESTION PAPER).*?(\b20\d{2}\b)", txt)
        if m:
            current_year = m.group(2)
            continue
        # detect class e.g., "Class XII" or "CLASS X"
        m2 = re.search(r"\bCLASS\s*(X{1,3}|[0-9]{1,2})\b", txt)
        if m2:
            current_class = tag.get_text(strip=True)
            continue

    # Now search for individual subject rows: many links are anchors or buttons labeled 'Download'
    # Find any anchor with href pointing to .pdf or .zip
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        href_full = urljoin(base_url, href)
        if href and re.search(r"\.(pdf|zip)(\?|$)", href, flags=re.I):
            # try to derive subject from nearby text
            # often the subject name is prior sibling or parent <td> or text
            subject_text = a.get_text(strip=True) or a.parent.get_text(" ", strip=True)
            downloads.append({
                "year": current_year,
                "class": current_class,
                "subject": subject_text or Path(href).stem,
                "url": href_full,
            })

    # JS-driven links: look for onclick attributes containing .pdf or .zip
    for el in soup.find_all(attrs={"onclick": True}):
        onclick = el.get("onclick")
        m = re.search(r"(https?:\/\/[^\s'\"\\)]+?\.(?:pdf|zip))", onclick, flags=re.I)
        if m:
            url = m.group(1)
            subject_text = el.get_text(strip=True) or "subject"
            downloads.append({
                "year": current_year,
                "class": current_class,
                "subject": subject_text,
                "url": url,
            })

    # fallback: search for any text tokens with direct .pdf/.zip urls
    for txt in soup.strings:
        m = re.search(r"(https?:\/\/[^\s'\"\\)]+?\.(?:pdf|zip))", txt, flags=re.I)
        if m:
            downloads.append({
                "year": current_year,
                "class": current_class,
                "subject": "unknown",
                "url": m.group(1),
            })

    # deduplicate by URL
    seen = set()
    deduped = []
    for d in downloads:
        if d["url"] in seen:
            continue
        seen.add(d["url"])
        deduped.append(d)
    return deduped


def main():
    session = get_session()

    if not allowed_by_robots(START_URL, user_agent="*"):
        logging.error(f"robots.txt disallows crawling {START_URL}. Exiting.")
        return

    logging.info(f"Fetching start page: {START_URL}")
    # first fetch page
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    r = session.get(START_URL, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text

    queue = parse_and_queue_downloads(html, START_URL)
    logging.info(f"Found {len(queue)} candidate downloads")

    for entry in queue:
        year = entry["year"] or "unknown_year"
        cls = entry["class"] or "unknown_class"
        subject = entry["subject"] or "unknown_subject"
        url = entry["url"]

        # build folder
        folder = OUTPUT_DIR / year / cls / re.sub(r"[^\w\-_ ]", "_", subject.strip())
        folder.mkdir(parents=True, exist_ok=True)

        filename = Path(urlparse(url).path).name
        if not filename:
            filename = make_filename_from_subject(subject, url)
        target_path = folder / filename

        if target_path.exists():
            logging.info(f"Already downloaded {target_path}, skipping.")
            continue

        polite_sleep()
        logging.info(f"Downloading {url} -> {target_path}")
        data, ctype = download_binary(session, url, referer=START_URL)
        if data is None:
            continue

        # If zip, extract PDFs
        if filename.lower().endswith(".zip") or (ctype and "zip" in ctype):
            saved = extract_pdfs_from_zip_bytes(data, folder)
            if saved:
                logging.info(f"Extracted {len(saved)} pdf(s) to {folder}")
            else:
                # maybe the zip contained non-pdf; save zip for inspection
                ztmp = folder / filename
                with open(ztmp, "wb") as f:
                    f.write(data)
                logging.info(f"Saved zip for inspection: {ztmp}")
        else:
            # ensure content looks like a PDF (first bytes %PDF)
            if data[:4] == b"%PDF":
                save_pdf_bytes(target_path, data)
                logging.info(f"Saved PDF: {target_path}")
            else:
                # maybe it's an HTML redirect or container that links to pdf; try to parse text
                text = data[:1024].decode("utf-8", errors="ignore")
                m = re.search(r"(https?:\/\/[^\s'\"\\)]+?\.pdf)", text, flags=re.I)
                if m:
                    pdf_url = m.group(1)
                    logging.info(f"Found embedded pdf link, downloading {pdf_url}")
                    polite_sleep()
                    pdata, _ = download_binary(session, pdf_url, referer=url)
                    if pdata and pdata[:4] == b"%PDF":
                        ptarget = folder / Path(urlparse(pdf_url).path).name
                        save_pdf_bytes(ptarget, pdata)
                        logging.info(f"Saved extracted PDF: {ptarget}")
                    else:
                        logging.warning(f"Failed to retrieve embedded pdf from {pdf_url}")
                else:
                    # save as .bin for inspection
                    bpath = folder / (filename + ".bin")
                    with open(bpath, "wb") as f:
                        f.write(data)
                    logging.warning(f"Saved unknown content to {bpath} for manual inspection")


if __name__ == "__main__":
    main()

