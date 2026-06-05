import sqlite3

c = sqlite3.connect('/app/data/aml.db')

print("=== TABLES ===")
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for t in tables:
    print(f"  - {t}")

print("\n=== TRANSACTIONS ===")
print(f"  Total:           {c.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]}")
print(f"  NULL risk_score: {c.execute('SELECT COUNT(*) FROM transactions WHERE risk_score IS NULL').fetchone()[0]}")
print(f"  Scored:          {c.execute('SELECT COUNT(*) FROM transactions WHERE risk_score IS NOT NULL').fetchone()[0]}")

print("\n=== BY STATUS ===")
for row in c.execute('SELECT status, COUNT(*) FROM transactions GROUP BY status'):
    print(f"  {row[0]:20s} {row[1]}")

print("\n=== ALERTS SCHEMA ===")
for row in c.execute("PRAGMA table_info(alerts)"):
    print(f"  {row[1]:25s} {row[2]}")

print("\n=== ALERTS COUNT ===")
print(f"  Total: {c.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]}")

print("\n=== TRANSACTIONS SCHEMA ===")
for row in c.execute("PRAGMA table_info(transactions)"):
    print(f"  {row[1]:25s} {row[2]}")