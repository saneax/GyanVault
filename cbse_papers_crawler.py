#!/usr/bin/env python3
"""
cbse_papers_crawler.py
A generic web crawler to download educational papers (PDFs/ZIPs).
It intelligently extracts metadata (year, class, subject) from URLs and saves files in a structured format.
It avoids re-downloading content by tracking file hashes (MD5).
"""
import os
import re
import io
import sys
import time
import json
import random
import sqlite3
import logging
import hashlib
import argparse
import zipfile
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

# ---------- CONFIG (Defaults, can be overridden by args) ----------
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS downloads (
                complete_url TEXT PRIMARY KEY,
                institution TEXT,
                type TEXT,
                year TEXT,
                class TEXT,
                subject TEXT,
                md5 TEXT,
                size INTEGER,
                path TEXT,
                content_type TEXT,
                etag TEXT,
                last_modified TEXT,
                pdfs_json TEXT,
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
    cur = conn.cursor()

    def existing_columns(table):
        cur.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    # downloads table expected columns
    expected_downloads = {
        "complete_url",
        "institution",
        "type",
        "year",
        "class",
        "subject",
        "md5",
        "size",
        "path",
        "content_type",
        "etag",
        "last_modified",
        "pdfs_json",
        "ts",
    }
    try:
        downloads_cols = existing_columns("downloads")
    except sqlite3.OperationalError:
        # missing downloads table -> create fresh
        logging.info("DB migration: creating downloads table")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS downloads (
                complete_url TEXT PRIMARY KEY,
                institution TEXT,
                type TEXT,
                year TEXT,
                class TEXT,
                subject TEXT,
                md5 TEXT,
                size INTEGER,
                path TEXT,
                content_type TEXT,
                etag TEXT,
                last_modified TEXT,
                pdfs_json TEXT,
                ts TEXT
            );
            """
        )
        conn.commit()
        downloads_cols = existing_columns("downloads")

    missing = expected_downloads - set(downloads_cols)
    for col in missing:
        # choose types sensibly (TEXT for most, INTEGER for size)
        typ = "INTEGER" if col == "size" else "TEXT"
        logging.info(f"DB migration: adding column '{col}' to downloads (type {typ})")
        cur.execute(f"ALTER TABLE downloads ADD COLUMN {col} {typ}")

    # md5_map table
    expected_md5map = {"md5", "path", "size", "ts"}
    try:
        md5map_cols = existing_columns("md5_map")
    except sqlite3.OperationalError:
        md5map_cols = []
    if not md5map_cols:
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
        missing2 = expected_md5map - set(md5map_cols)
        for col in missing2:
            typ = "INTEGER" if col == "size" else "TEXT"
            logging.info(f"DB migration: adding column '{col}' to md5_map (type {typ})")
            cur.execute(f"ALTER TABLE md5_map ADD COLUMN {col} {typ}")

    conn.commit()


def db_get_download(conn, url: str):
    cur = conn.cursor()
    cur.execute("SELECT complete_url, institution, type, year, class, subject, md5, size, path, content_type, etag, last_modified, pdfs_json, ts FROM downloads WHERE complete_url = ?", (url,))
    row = cur.fetchone()
    if row:
        keys = ["complete_url", "institution", "type", "year", "class", "subject", "md5", "size", "path", "content_type", "etag", "last_modified", "pdfs_json", "ts"]
        res = dict(zip(keys, row))
        # parse pdfs_json into list
        if res.get("pdfs_json"):
            try:
                res["pdfs"] = json.loads(res["pdfs_json"])
            except Exception:
                res["pdfs"] = []
        else:
            res["pdfs"] = []
        return res
    return None


def db_insert_download(conn, complete_url: str, institution: str, typ: str, year: str, cls: str, subject: str, md5: str, size: int, path: str, content_type: str, pdfs: list, etag: str = None, last_modified: str = None):
    cur = conn.cursor()
    pdfs_json = json.dumps(pdfs or [])
    cur.execute(
        """
        INSERT OR REPLACE INTO downloads(complete_url, institution, type, year, class, subject, md5, size, path, content_type, etag, last_modified, pdfs_json, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (complete_url, institution or "", typ or "", year or "", cls or "", subject or "", md5 or "", size or 0, path or "", content_type or "", etag or "", last_modified or "", pdfs_json, datetime.utcnow().isoformat()),
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


def dissect_url(url: str):
    """
    Break URL path into institution, type, year, class, subject (best-effort).
    Example:
      https://some-site.edu/papers/2024/Class-XII/Physics.pdf
    -> institution=some-site.edu, type=papers, year=2024, class=Class-XII, subject=Physics.pdf
    """
    parsed = urlparse(url)
    parts = [seg for seg in parsed.path.split("/") if seg]
    
    # Institution is derived from the domain name for better generality
    institution = parsed.netloc.replace("www.", "")
    typ = parts[1] if len(parts) >= 2 else ""
    year = None
    cls = None
    subject = None

    # More generic regex to find year and class in URL segments
    for i, seg in enumerate(parts):
        m = re.match(r"^(20\d{2})", seg)
        if m:
            year = m.group(1)
            # Check if the next segment looks like a class/grade
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if re.match(r"^(?:X{1,3}|I{1,3}V?|IV|V|VI{1,3}|IX|XI{0,2}|[0-9]{1,2}|Class-?[0-9]{1,2})$", nxt, flags=re.I):
                    cls = nxt

    # subject fallback: last path part or filename
    if parts:
        subject = parts[-1]
    return {
        "institution": institution,
        "type": typ,
        "year": year or "",
        "class": cls or "",
        "subject": subject or "",
    }


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
            parts = dissect_url(href_full)
            downloads.append({
                "url": normalize_url(href_full),
                "subject": sanitize(subj),
                "year": parts.get("year") or current_year or "",
                "class": parts.get("class") or current_class or "",
                "institution": parts.get("institution") or "",
                "type": parts.get("type") or "",
            })
    # onclick JS with full URLs
    for el in soup.find_all(attrs={"onclick": True}):
        onclick = el.get("onclick") or ""
        m = re.search(r"(https?:\/\/[^\s'\"\\)]+?\.(?:pdf|zip))", onclick, flags=re.I)
        if m:
            u = normalize_url(m.group(1))
            subj = el.get_text(" ", strip=True) or "subject"
            parts = dissect_url(u)
            downloads.append({
                "url": u,
                "subject": sanitize(subj),
                "year": parts.get("year") or current_year or "",
                "class": parts.get("class") or current_class or "",
                "institution": parts.get("institution") or "",
                "type": parts.get("type") or "",
            })
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
    parser = argparse.ArgumentParser(description="A generic crawler for academic papers.")
    parser.add_argument("start_url", type=str, help="The starting URL to crawl for papers.")
    args = parser.parse_args()

    start_url = args.start_url

    conn = init_db(DB_PATH)
    session = get_session()
    TEMP_DIR.mkdir(exist_ok=True)

    if not allowed_by_robots(start_url):
        logging.error("robots.txt disallows crawling the start URL. Exiting.")
        return

    headers = {"User-Agent": random.choice(USER_AGENTS)}
    logging.info(f"Fetching start page: {start_url}")
    r = session.get(start_url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    html = r.text

    queue = parse_and_queue_downloads(html, start_url)
    logging.info(f"Found {len(queue)} candidate downloads")

    for idx, item in enumerate(queue, 1):
        url = item["url"]
        subject_from_item = item.get("subject") or ""
        year_from_item = item.get("year") or ""
        class_from_item = item.get("class") or ""
        inst = item.get("institution") or ""
        typ = item.get("type") or ""

        logging.info(f"[{idx}/{len(queue)}] Processing: {url}")

        # check existing
        rec = db_get_download(conn, url)
        if rec:
            logging.info(f"Already downloaded: {url} -> {rec.get('path')} (skipping)")
            continue

        # HEAD info (etag)
        head = head_check(session, url, referer=start_url)
        if head and head.get("etag"):
            logging.debug(f"HEAD ETag: {head.get('etag')} for {url}")

        polite_sleep()

        # compute structured fields from URL (defensive)
        institution = inst or dissected.get("institution") or ""
        typ_field = typ or dissected.get("type") or ""
        year = year_from_item or dissected.get("year") or ""
        cls = class_from_item or dissected.get("class") or ""
        subject = subject_from_item or dissected.get("subject") or dissected.get("subject") or ""

        folder = OUTPUT_DIR / (year or "unknown_year") / (cls or "unknown_class") / sanitize(subject)
        folder.mkdir(parents=True, exist_ok=True)

        fname = Path(urlparse(url).path).name or (sanitize(subject) + ".pdf")
        temp_path = TEMP_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".tmp")

        info = stream_download_to_temp(session, url, temp_path, referer=start_url)
        if not info:
            logging.warning(f"Failed to download {url}")
            continue

        file_md5 = info["md5"]
        size = info["size"]
        ctype = info.get("content_type", "")

        # if content already exists by md5, don't duplicate; just reference existing path
        md5rec = db_get_md5_map(conn, file_md5)
        if md5rec:
            existing_path = Path(md5rec["path"])
            logging.info(f"Content already exists (md5 match). Not duplicating. Using {existing_path}")
            # record download metadata pointing to existing file and empty pdfs_json (we can fill later)
            db_insert_download(conn, url, institution, typ_field, year, cls, subject, file_md5, size, str(existing_path), ctype, pdfs=[], etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
            try:
                temp_path.unlink()
            except Exception:
                pass
            continue

        is_zip = fname.lower().endswith(".zip") or ("zip" in (ctype or "").lower())

        if is_zip:
            # move temp file into place as the zip
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

            # extract PDFs
            saved_pdfs = extract_pdfs_from_zip(zpath, folder)
            pdfs_meta = []
            if saved_pdfs:
                logging.info(f"Extracted {len(saved_pdfs)} PDFs from {zpath} -> {folder}. Deleting zip.")
                for p in saved_pdfs:
                    try:
                        p_md5 = md5_of_file(p)
                        p_size = p.stat().st_size
                        db_insert_md5_map(conn, p_md5, str(p), p_size)
                        pdfs_meta.append({"file": str(p.relative_to(OUTPUT_DIR)), "md5": p_md5, "size": p_size})
                    except Exception as e:
                        logging.warning(f"Failed to record md5 for {p}: {e}")
                        pdfs_meta.append({"file": str(p.relative_to(OUTPUT_DIR)), "md5": None, "size": p.stat().st_size if p.exists() else None})
                # record the zip's md5 and that it was extracted
                db_insert_md5_map(conn, file_md5, str(zpath), size)
                db_insert_download(conn, url, institution, typ_field, year, cls, subject, file_md5, size, "zip_extracted_and_deleted", "application/zip", pdfs_meta, etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
                # delete zip to save space
                try:
                    zpath.unlink()
                except Exception:
                    pass
            else:
                # no PDFs inside: keep zip and record it; pdfs_meta is empty
                logging.info(f"No PDFs found in zip; kept zip at {zpath}")
                db_insert_md5_map(conn, file_md5, str(zpath), size)
                db_insert_download(conn, url, institution, typ_field, year, cls, subject, file_md5, size, str(zpath), "application/zip", pdfs_meta, etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
            continue

        # not a zip -- likely a single pdf or other file
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

        # compute actual md5 of saved file (safety)
        try:
            actual_md5 = md5_of_file(target)
        except Exception:
            actual_md5 = file_md5

        # store md5_map and a single-entry pdfs list if it's a PDF
        db_insert_md5_map(conn, actual_md5, str(target), target.stat().st_size if target.exists() else size)
        pdfs_meta = []
        if target.exists() and target.suffix.lower() == ".pdf":
            pdfs_meta.append({"file": str(target.relative_to(OUTPUT_DIR)), "md5": actual_md5, "size": target.stat().st_size})
        else:
            # non-pdf single file; still record as pdfs_meta empty or with file
            pdfs_meta.append({"file": str(target.relative_to(OUTPUT_DIR)), "md5": actual_md5, "size": target.stat().st_size if target.exists() else size})

        db_insert_download(conn, url, institution, typ_field, year, cls, subject, actual_md5, target.stat().st_size if target.exists() else size, str(target), ctype or "application/octet-stream", pdfs_meta, etag=head.get("etag") if head else None, last_modified=head.get("last_modified") if head else None)
        logging.info(f"Saved file: {target}")

    conn.close()
    logging.info("All done.")


if __name__ == "__main__":
    main()
