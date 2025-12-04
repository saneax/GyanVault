import sqlite3
import os

# Ensure we are opening the correct file
db_path = "downloads.db"
print(f"Connecting to: {os.path.abspath(db_path)}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # 1. SANITY CHECK: Do we have ANY data?
    cursor.execute("SELECT count(*) FROM downloads")
    total = cursor.fetchone()[0]
    print(f"Total rows in DB: {total}")
    
    if total == 0:
        print("!! ALERT: The database is empty.")
    else:
        # 2. CASE SENSITIVITY CHECK: Force lowercase search
        # We use LOWER() to ignore case completely.
        print("\nAttempting robust search...")
        
        query = """
            SELECT count(*)
            FROM downloads 
            WHERE 
                LOWER(path) LIKE '%mathematics%' 
                OR 
                LOWER(pdfs_json) LIKE '%mathematics%'
        """
        
        cursor.execute(query)
        match_count = cursor.fetchone()[0]
        print(f"Matches found (using LOWER + Hardcoded pattern): {match_count}")

        if match_count > 0:
            print("\nFetching first 5 matches:")
            fetch_query = """
                SELECT path, pdfs_json 
                FROM downloads 
                WHERE 
                    LOWER(path) LIKE '%mathematics%' 
                    OR 
                    LOWER(pdfs_json) LIKE '%mathematics%'
                LIMIT 5
            """
            cursor.execute(fetch_query)
            for row in cursor.fetchall():
                print(f"Match found in: {row[0] if 'mathematics' in row[0].lower() else 'JSON content'}")

except Exception as e:
    print(f"!! Error: {e}")
finally:
    conn.close()