import sqlite3 as sq

try:
    db = sq.connect("qqq.db", timeout=5)
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE student (id INTEGER,
                                            name TEXT,
                                            born_date DATE);''')
    cursor.close()
except sq.Error as error:
    print("Ошибка")

if db:
    db.close()
    print("база данных закрыта")
