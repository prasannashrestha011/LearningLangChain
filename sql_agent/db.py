import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("agent.db")
cursor = conn.cursor()

# Create 'employees' table
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    position TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL,
    hire_date TEXT
)
"""
)

# Insert some dummy data
employees_data = [
    (1, "Alice Smith", "Software Engineer", "IT", 75000, "2022-01-15"),
    (2, "Bob Johnson", "Product Manager", "Product", 85000, "2021-09-01"),
    (3, "Charlie Lee", "Data Analyst", "Analytics", 65000, "2023-03-10"),
    (4, "Diana King", "HR Specialist", "HR", 60000, "2020-07-23"),
    (5, "Ethan Brown", "DevOps Engineer", "IT", 78000, "2022-11-05"),
]

cursor.executemany(
    """
INSERT OR REPLACE INTO employees (id, name, position, department, salary, hire_date)
VALUES (?, ?, ?, ?, ?, ?)
""",
    employees_data,
)

# Commit changes and close connection
conn.commit()
conn.close()

print("Dummy employees table created with sample data.")
