import psycopg2

def get_pg_connection():
    return psycopg2.connect(
        host="dpg-d18e21ruibrs73bqcc7g-a",
        dbname="lungdata",
        user="lungdata_user",
        password="cSTm86aPOICCcnolOFWixkVkTd77vVk6"
    )
