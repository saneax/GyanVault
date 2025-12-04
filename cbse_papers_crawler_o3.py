#!/usr/bin/env python3
"""
cbse_papers_crawler.py
Robust CBSE question-paper crawler.
 - Normalizes URLs and avoids re-downloading
 - SQLite DB with downloads and md5_map
 - Migrates old DB schemas (adds missing columns if required)
 - HEAD/Etag support and streamed downloads with progress
 - Extract PDFs from ZIPs and delete ZIPs after extraction
 - Folder layout: output/{year}/{class}/{subject}/{file}
Run: python cbse_papers_crawler.py
"""

import os
import re
import io
import sys
import time
import random
import sqlite3
import logging
import hashlib
import zipfile
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

# ---------- CONFIG ----------
START_URL = "https://www.cbse.gov.in/cbsenew/question-paper.html"
OUTPUT_DIR = Path("output")
DB_PATH = Path("downloads.db")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

MIN_DELAY = 2.5
MAX_DELAY = 9.5
TIMEOUT = 30
MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 32
TEMP_DIR = Path(".temp_downloads")
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def polite_sleep():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def normalize_url(url: str) -> str:
    u, _ = urldefrag(url)
    p = urlparse(u)
    new = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower())
    return urlunparse(new)


# ---------- DB helpers & migration ----------
def init_db(path: Path):
    created = not path.exists()
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    if created:
        # create fresh tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS downloads (
                url TEXT PRIMARY KEY,
                md5 TEXT,
                size INTEGER,
                path TEXT,
                content_type TEXT,
                etag TEXT,
                last_modified TEXT,
                ts TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md5_map (
                md5 TEXT PRIMARY KEY,
                path TEXT,
                size INTEGER,
                ts TEXT
            );
            """
        )
        conn.commit()
    else:
        migrate_db(conn)
    return conn


def migrate_db(conn: sqlite3.Connection):
    # Ensure required columns exist in downloads and md5_map; add missing columns.
    cur = conn.cursor()

    def existing_columns(table):
        cur.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    downloads_cols = existing_columns("downloads")
    expected_downloads = {"url", "md5", "size", "path", "content_type", "etag", "last_modified", "ts"}
    missing = expected_downloads - set(downloads_cols)
    for col in missing:
        # choose types sensibly
        if col in ("md5", "path", "content_type", "etag", "last_modified", "ts"):
            typ = "TEXT"
        else:
            typ = "INTEGER"
        logging.info(f"DB migration: adding column '{col}' to downloads (type {typ})")
        cur.execute(f"ALTER TABLE downloads ADD COLUMN {col} {typ}")
    # md5_map
    try:
        md5map_cols = existing_columns("md5_map")
    except sqlite3.OperationalError:
        md5map_cols = []
    expected_md5map = {"md5", "path", "size", "ts"}
    missing2 = expected_md5map - set(md5map_cols)
    if not md5map_cols:
        # table doesn't exist, create it
        logging.info("DB migration: creating md5_map table")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS md5_map (
                md5 TEXT PRIMARY KEY,
                path TEXT,
                size INTEGER,
                ts TEXT
            );
            """
        )
    else:
        for col in missing2:
            typ = "TEXT" if col in ("md5", "path", "ts") else "INTEGER"
            logging.info(f"DB migration: adding column '{col}' to md5_map (type {typ})")
            cur.execute(f"ALTER TABLE md5_map ADD COLUMN {col} {typ}")
    conn.commit()


def db_get_download(conn, url: str):
    cur = conn.cursor()
    cur.execute("SELECT url, md5, size, path, content_type, etag, last_modified, ts FROM downloads WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        keys = ["url", "md5", "size", "path", "content_type", "etag", "last_modified", "ts"]
        return dict(zip(keys, row))
    return None


def db_insert_download(conn, url, md5, size, path, content_type, etag=None, last_modified=None):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO downloads(url, md5, size, path, content_type, etag, last_modified, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (url, md5, size, path, content_type or "", etag or "", last_modified or "", datetime.utcnow().isoformat()),
    )
    conn.commit()


def db_get_md5_map(conn, md5: str):
    cur = conn.cursor()
    cur.execute("SELECT md5, path, size, ts FROM md5_map WHERE md5 = ?", (md5,))
    row = cur.fetchone()
    if row:
        keys = ["md5", "path", "size", "ts"]
        return dict(zip(keys, row))
    return None


def db_insert_md5_map(conn, md5: str, path: str, size: int):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO md5_map(md5, path, size, ts) VALUES (?, ?, ?, ?)", (md5, path, size, datetime.utcnow().isoformat()))
    conn.commit()


# ---------- utils ----------
def md5_of_file(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b: bytes):
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def sanitize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\/\\\:\*\?\"<>\|]+", "_", s)
    s = re.sub(r"\s+", " ", s)
    return s or "unknown"


def infer_year_class_from_url(url: str):
    p = urlparse(url)
    parts = [seg for seg in p.path.split("/") if seg]
    for i, seg in enumerate(parts):
        m = re.match(r"^(20\d{2})", seg)
        if m:
            year = m.group(1)
            cls = None
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if re.match(r"^(?:X{1,3}|I|II|III|IV|V|VI|VII|VIII|IX|XI|XII|[0-9]{1,2}|Class[_\- ]?[0-9]{1,2})$", nxt, flags=re.I):
                    cls = nxt
            return year, cls
    return None, None


# ---------- HTTP ----------
def get_session():
    s = requests.Session()
    s.headers.update({"Accept-Language": "en-US,en;q=0.9"})
    return s


def allowed_by_robots(url: str, ua: str = "*"):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        logging.warning("Couldn't fetch robots.txt; proceeding but staying polite")
        return True


def head_check(session: requests.Session, url: str, referer: str = None):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    if referer:
        headers["Referer"] = referer
    try:
        r = session.head(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        if 200 <= r.status_code < 400:
            return {
                "status_code": r.status_code,
                "content_length": int(r.headers.get("Content-Length", "0") or 0),
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "content_type": r.headers.get("Content-Type", ""),
            }
    except Exception as e:
        logging.debug(f"HEAD failed for {url}: {e}")
    return None


def stream_download_to_temp(session: requests.Session, url: str, temp_path: Path, referer: str = None):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    if referer:
        headers["Referer"] = referer
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0) or 0)
                h = hashlib.md5()
                size = 0
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, leave=False, desc=temp_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                        pbar.update(len(chunk))
                return {"md5": h.hexdigest(), "size": size, "content_type": r.headers.get("Content-Type", "")}
        except Exception as e:
            logging.warning(f"Download attempt {attempt} failed for {url}: {e}")
            time.sleep(1 + attempt * 2)
    return None


# ---------- parsing ----------
def parse_and_queue_downloads(page_html: str, base_url: str):
    soup = BeautifulSoup(page_html, "html.parser")
    downloads = []
    current_year = None
    current_class = None

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "p", "div"]):
        txt = (tag.get_text(" ", strip=True) or "").upper()
        m = re.search(r"(EXAMINATION|QUESTION PAPER).*?(\b20\d{2}\b)", txt)
        if m:
            current_year = m.group(2)
        m2 = re.search(r"\bCLASS\s*(X{1,3}|[0-9]{1,2})\b", txt, flags=re.I)
        if m2:
            current_class = m2.group(0)

    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if not href:
            continue
        href_full = urljoin(base_url, href)
        if re.search(r"\.(pdf|zip)(\?|$)", href, flags=re.I):
            subj = a.get_text(" ", strip=True) or a.parent.get_text(" ", strip=True) or Path(href).stem
            y, c = infer_year_class_from_url(href_full)
            downloads.append({"url": normalize_url(href_full), "subject": sanitize(subj), "year": y or current_year or "unknown_year", "cls": c or current_class or "unknown_class"})
    for el in soup.find_all(attrs={"onclick": True}):
        onclick = el.get("onclick") or ""
        m = re.search(r"(https?:\/\/[^\s'\"\\)]+?\.(?:pdf|zip))", onclick, flags=re.I)
        if m:
            u = normalize_url(m.group(1))
            subj = el.get_text(" ", strip=True) or "subject"
            y, c = infer_year_class_from_url(u)
            downloads.append({"url": u, "subject": sanitize(subj), "year": y or current_year or "unknown_year", "cls": c or current_class or "unknown_class"})
    seen = set()
    out = []
    for d in downloads:
        if d["url"] in seen:
            continue
        seen.add(d["url"])
        out.append(d)
    return out


# ---------- file ops ----------
def extract_pdfs_from_zip(zip_path: Path, target_folder: Path):
    saved = []
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for zi in z.infolist():
                if zi.filename.lower().endswith(".pdf"):
                    name = Path(zi.filename).name
                    out = target_folder / sanitize(name)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(zi) as src, open(out, "wb") as dst:
                        dst.write(src.read())
                    saved.append(out)
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_path}")
    return saved


# ---------- main ----------
def main():
    conn = init_db(DB_PATH)
    session = get_session()
    TEMP_DIR.mkdir(exist_ok=True)

    if not allowed_by_robots(START_URL):
        logging.error("robots.txt disallows crawling the start URL. Exiting.")
        return

    headers = {"User-Agent": random.choice(USER_AGENTS)}
    logging.info(f"Fetching start page: {START_URL}")
    r = session.get(START_URL, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text

    queue = parse_and_queue_downloads(html, START_URL)
    logging.info(f"Found {len(queue)} candidate downloads")

    for idx, item in enumerate(queue, 1):
        url = item["url"]
        subject = sanitize(item.get("subject"))
        year = sanitize(str(item.get("year")))
        cls = sanitize(str(item.get("cls")))

        logging.info(f"[{idx}/{len(queue)}] Processing: {url}")
        rec = db_get_download(conn, url)
        if rec:
            logging.info(f"Already downloaded: {url} -> {rec.get('path')} (skipping)")
            continue

        head = head_check(session, url, referer=START_URL)
        if head and head.get("etag"):
            logging.debug(f"HEAD ETag: {head.get('etag')} for {url}")

        polite_sleep()

        folder = OUTPUT_DIR / year / cls / subject
        folder.mkdir(parents=True, exist_ok=True)

        fname = Path(urlparse(url).path).name or (subject + ".pdf")
        temp_path = TEMP_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".tmp")

        info = stream_download_to_temp(session, url, temp_path, referer=START_URL)
        if not info:
            logging.warning(f"Failed to download {url}")
            continue

        file_md5 = info["md5"]
        size = info["size"]
        ctype = info.get("content_type", "")

        md5rec = db_get_md5_map(conn, file_md5)
        if md5rec:
            existing_path = Path(md5rec["path"])
            logging.info(f"Content already exists (md5 match). Not duplicating. Using {existing_path}")
            db_insert_download(conn, url, file_md5, size, str(existing_path), ctype, etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
            try:
                temp_path.unlink()
            except Exception:
                pass
            continue

        is_zip = fname.lower().endswith(".zip") or ("zip" in (ctype or "").lower())
        if is_zip:
            zpath = folder / fname
            try:
                tmpz = zpath.with_suffix(zpath.suffix + ".part")
                temp_path.replace(tmpz)
                tmpz.replace(zpath)
            except Exception:
                try:
                    with open(temp_path, "rb") as src, open(zpath, "wb") as dst:
                        dst.write(src.read())
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                except Exception as e:
                    logging.error(f"Failed saving zip to {zpath}: {e}")
                    continue

            saved_pdfs = extract_pdfs_from_zip(zpath, folder)
            if saved_pdfs:
                logging.info(f"Extracted {len(saved_pdfs)} PDFs from {zpath} -> {folder}. Deleting zip.")
                for p in saved_pdfs:
                    try:
                        p_md5 = md5_of_file(p)
                        db_insert_md5_map(conn, p_md5, str(p), p.stat().st_size)
                    except Exception as e:
                        logging.warning(f"Failed to record md5 for {p}: {e}")
                db_insert_download(conn, url, file_md5, size, "zip_extracted_and_deleted", "application/zip", etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
                try:
                    zpath.unlink()
                except Exception:
                    pass
            else:
                logging.info(f"No PDFs found in zip; kept zip at {zpath}")
                db_insert_md5_map(conn, file_md5, str(zpath), size)
                db_insert_download(conn, url, file_md5, size, str(zpath), "application/zip", etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
            continue

        target = folder / fname
        try:
            tmp_target = target.with_suffix(target.suffix + ".part")
            temp_path.replace(tmp_target)
            tmp_target.replace(target)
        except Exception:
            try:
                with open(temp_path, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"Failed saving file to {target}: {e}")
                continue

        try:
            actual_md5 = md5_of_file(target)
        except Exception:
            actual_md5 = file_md5

        db_insert_md5_map(conn, actual_md5, str(target), target.stat().st_size)
        db_insert_download(conn, url, actual_md5, target.stat().st_size, str(target), ctype or "application/octet-stream", etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
        logging.info(f"Saved file: {target}")

    conn.close()
    logging.info("All done.")


if __name__ == "__main__":
    main()

