import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class DBManager:
    """A manager for interacting with the 'downloads.db' SQLite database."""

    def __init__(self, db_path="downloads.db"):
        """
        Initializes the DBManager, connects to the database, and ensures the
        schema is up-to-date.
        """
        self.db_path = Path(db_path)
        self.conn = self._init_db()
        if self.conn:
            self.conn.row_factory = self._dict_factory

    def _dict_factory(self, cursor, row):
        """Converts query results from tuples to dictionaries."""
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    def _init_db(self):
        """
        Initializes the database connection. Creates the DB and tables if they
        don't exist, and runs schema migrations if they do.
        """
        created = not self.db_path.exists()
        try:
            conn = sqlite3.connect(str(self.db_path))
            if created:
                logging.info(f"Creating new database at: {self.db_path}")
                self._create_tables(conn)
            else:
                self._migrate_db(conn)
            return conn
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            return None

    def _create_tables(self, conn):
        """Creates the initial 'downloads' and 'md5_map' tables."""
        cur = conn.cursor()
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

    def _migrate_db(self, conn):
        """
        Checks for missing columns in existing tables and adds them, ensuring
        backward compatibility with older database schemas.
        """
        cur = conn.cursor()
        
        def get_existing_columns(table_name):
            try:
                cur.execute(f"PRAGMA table_info({table_name})")
                return {row[1] for row in cur.fetchall()}
            except sqlite3.OperationalError:
                return set()

        # Check 'downloads' table
        expected_downloads_cols = {
            "complete_url", "institution", "type", "year", "class", "subject",
            "md5", "size", "path", "content_type", "etag", "last_modified",
            "pdfs_json", "ts"
        }
        downloads_cols = get_existing_columns("downloads")
        if not downloads_cols: # Table doesn't exist
            self._create_tables(conn)
            downloads_cols = get_existing_columns("downloads")

        missing_cols = expected_downloads_cols - downloads_cols
        for col in missing_cols:
            col_type = "INTEGER" if col == "size" else "TEXT"
            logging.info(f"DB Migration: Adding column '{col}' to 'downloads' table.")
            cur.execute(f"ALTER TABLE downloads ADD COLUMN {col} {col_type}")

        conn.commit()

    def close(self):
        """Closes the database connection if it's open."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

    def add_or_update_download(self, **kwargs):
        """
        Adds a new download record or replaces an existing one based on the
        'complete_url' primary key. Accepts keyword arguments matching the
        table schema.
        """
        if 'complete_url' not in kwargs:
            raise ValueError("'complete_url' is a required argument.")

        # Ensure all fields are present, defaulting to None or an empty value
        fields = [
            "complete_url", "institution", "type", "year", "class", "subject",
            "md5", "size", "path", "content_type", "etag", "last_modified",
            "pdfs_json", "ts"
        ]
        
        # Set timestamp for new/updated record
        kwargs['ts'] = datetime.utcnow().isoformat()
        
        # Convert pdfs list to json string if it's a list
        if 'pdfs_json' in kwargs and isinstance(kwargs['pdfs_json'], list):
            kwargs['pdfs_json'] = json.dumps(kwargs['pdfs_json'])

        values = [kwargs.get(field) for field in fields]
        
        sql = f"""
            INSERT OR REPLACE INTO downloads ({', '.join(fields)})
            VALUES ({', '.join(['?'] * len(fields))})
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(sql, values)
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Failed to add/update record for {kwargs['complete_url']}: {e}")
            return None

    def get_download_by_url(self, url):
        """Fetches a single download record by its primary key (complete_url)."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM downloads WHERE complete_url = ?", (url,))
        return cur.fetchone()

    def update_record(self, url, updates):
        """
        Updates specific fields for a record identified by its URL.
        :param url: The complete_url of the record to update.
        :param updates: A dictionary of {'column_name': new_value}.
        """
        if not updates or not isinstance(updates, dict):
            logging.warning("Update called with no fields to update.")
            return False

        # Add a timestamp to the update
        updates['ts'] = datetime.utcnow().isoformat()

        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [url]
        
        sql = f"UPDATE downloads SET {set_clause} WHERE complete_url = ?"
        
        try:
            cur = self.conn.cursor()
            cur.execute(sql, values)
            self.conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"Failed to update record for {url}: {e}")
            return False

    def search(self, limit=None, offset=None, **kwargs):
        """
        Searches for records using keyword arguments for any field.
        Supports LIKE for string searches by including '%' in the value.
        
        Example: db.search(subject='%Math%', year='2023', limit=10)
        """
        where_clauses = []
        params = []

        for key, value in kwargs.items():
            # Use LIKE for strings, = for other types
            if isinstance(value, str) and '%' in value:
                where_clauses.append(f"LOWER({key}) LIKE ?")
                params.append(value.lower())
            else:
                where_clauses.append(f"{key} = ?")
                params.append(value)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"SELECT * FROM downloads WHERE {where_sql}"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)
            
        try:
            cur = self.conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Search failed: {e}")
            return []

if __name__ == "__main__":
    # Self-test cases for the DBManager module
    
    import os

    TEST_DB_PATH = "test_downloads.db"

    def run_tests():
        """Runs a sequence of tests on the DBManager."""
        print("--- Running DBManager Self-Tests ---")

        # 1. Initialization and Cleanup
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        
        db = DBManager(db_path=TEST_DB_PATH)
        if not db.conn:
            print("[FAIL] DBManager failed to initialize.")
            return

        print("[PASS] Initialization")

        # 2. Add a record
        url1 = "http://example.com/paper1.pdf"
        db.add_or_update_download(
            complete_url=url1,
            institution="CBSE",
            type="Question Paper",
            year="2023",
            class="XII",
            subject="Physics",
            path="output/2023/XII/Physics/paper1.pdf"
        )
        
        # 3. Get the record
        record1 = db.get_download_by_url(url1)
        assert record1 is not None and record1['subject'] == 'Physics'
        print("[PASS] Add and Get Record")

        # 4. Update the record
        db.update_record(url1, {"subject": "Physics_Updated"})
        record1_updated = db.get_download_by_url(url1)
        assert record1_updated['subject'] == 'Physics_Updated'
        print("[PASS] Update Record")

        # 5. Add a second record for searching
        url2 = "http://example.com/paper2.pdf"
        db.add_or_update_download(
            complete_url=url2,
            institution="CBSE",
            type="Marking Scheme",
            year="2023",
            class="XII",
            subject="Chemistry",
            path="output/2023/XII/Chemistry/paper2.pdf"
        )

        # 6. Search for records
        # Search by exact match
        physics_results = db.search(subject="Physics_Updated")
        assert len(physics_results) == 1 and physics_results[0]['complete_url'] == url1
        print("[PASS] Search (Exact Match)")

        # Search with multiple criteria
        cbse_2023_results = db.search(institution="CBSE", year="2023")
        assert len(cbse_2023_results) == 2
        print("[PASS] Search (Multiple Criteria)")

        # Search with LIKE
        chem_results = db.search(subject="%chem%")
        assert len(chem_results) == 1 and chem_results[0]['complete_url'] == url2
        print("[PASS] Search (LIKE)")

        # Search with limit
        limited_results = db.search(institution="CBSE", limit=1)
        assert len(limited_results) == 1
        print("[PASS] Search (Limit)")

        # 7. Test INSERT OR REPLACE
        db.add_or_update_download(
            complete_url=url1,
            institution="ICSE", # Changed institution
            subject="Physics_Replaced"
        )
        record1_replaced = db.get_download_by_url(url1)
        assert record1_replaced['institution'] == 'ICSE' and record1_replaced['subject'] == 'Physics_Replaced'
        all_records = db.search()
        assert len(all_records) == 2 # Ensure it replaced, not added
        print("[PASS] Add or Update (Replace)")

        # 8. Cleanup
        db.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        print("[PASS] Cleanup")
        print("\n--- All tests completed successfully! ---")

    try:
        run_tests()
    except AssertionError as e:
        print(f"\n[TEST FAIL] Assertion failed: {e}")
    except Exception as e:
        print(f"\n[CRITICAL FAIL] An unexpected error occurred: {e}")