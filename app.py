from flask import Flask, render_template, request, redirect, session, flash
from database_config import get_pg_connection
import re
import os
import pandas as pd
from model import classify_patient_condition
from cxr_model import predict_cxr

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ HOME PAGE ------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------ REGISTER ------------------
@app.route('/register', methods=['GET', 'POST'])   
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email'].strip().lower()
        mobile = request.form['mobile']
        password = request.form['password']
        user_type = request.form['user_type']

        if user_type not in ['user', 'admin']:
            flash('Please select a valid role.', 'danger')
            return redirect('/register')

        if not re.match(r'^[A-Za-z\s]{2,50}$', name):
            flash('Name should only contain letters and spaces.', 'danger')
            return redirect('/register')

        if not re.match(r'^[a-z][a-z0-9._%+-]*@gmail\.com$', email):
            flash('Only valid Gmail addresses are allowed.', 'danger')
            return redirect('/register')

        if not re.match(r'^[6-9][0-9]{9}$', mobile):
            flash('Enter a valid 10-digit Indian mobile number.', 'danger')
            return redirect('/register')

        if not re.match(r'^(?=.*[A-Z])(?=.*[^A-Za-z0-9]).{6,}$', password):
            flash('Password must contain at least one uppercase letter, one special character, and be at least 6 characters long.', 'danger')
            return redirect('/register')

        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            conn.close()
            flash('User already registered with this email.', 'warning')
            return redirect('/register')

        cursor.execute("INSERT INTO users (name, email, mobile, password, user_type) VALUES (%s, %s, %s, %s, %s)",
                       (name, email, mobile, password, user_type))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Registered successfully! Please login.', 'success')
        return redirect('/login_user' if user_type == 'user' else '/login_admin')

    return render_template('register.html')

# ------------------ LOGIN (USER) ------------------
@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s AND user_type='user'", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user:
            if user[4] == password:
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                return redirect('/user_home')
            else:
                error = "Incorrect password. Please try again."
        else:
            error = "Email not found."

    return render_template('login_user.html', error=error)

# ------------------ LOGIN (ADMIN) ------------------
@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND user_type = 'admin'", (email,))
        admin = cursor.fetchone()
        cursor.close()
        conn.close()

        if admin:
            if admin[4] == password:
                session['admin_id'] = admin[0]
                session['admin_name'] = admin[1]
                return redirect('/admin_dashboard')
            else:
                error = "Incorrect password. Please try again."
        else:
            error = "Admin account not found."

    return render_template('login_admin.html', error=error)

# ------------------ FORGOT PASSWORD ------------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        new_password = request.form['new_password']

        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, email))
            conn.commit()
            cursor.close()
            conn.close()
            return redirect('/login_user')
        else:
            cursor.close()
            conn.close()
            return render_template('forgot_password.html', error="Email not found.")

    return render_template('forgot_password.html')

# ------------------ USER HOME ------------------
@app.route('/user_home', methods=['GET', 'POST'])
def user_home():
    if 'user_id' not in session:
        return redirect('/login_user')

    name = session.get('user_name', 'User')
    user_id = session['user_id']
    cxr_result = vitals_result = None
    graph_data = []
    image_filename = None
    latest_report = None
    report_message = ""

    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT disease, vital_trend, uploaded_at
        FROM diagnosis_results
        WHERE user_id = %s
        ORDER BY uploaded_at DESC
        LIMIT 1
    """, (user_id,))
    latest_report = cursor.fetchone()
    cursor.close()
    conn.close()

    if not latest_report:
        report_message = "No previous report found."

    if request.method == 'POST':
        if 'cxr' in request.files:
            cxr = request.files['cxr']
            if cxr.filename != '':
                cxr_path = os.path.join(UPLOAD_FOLDER, cxr.filename)
                cxr.save(cxr_path)
                cxr_result = predict_cxr(cxr_path)
                image_filename = os.path.join('uploads', cxr.filename)

        if 'vitals' in request.files:
            csv = request.files['vitals']
            if csv.filename != '':
                csv_path = os.path.join(UPLOAD_FOLDER, csv.filename)
                csv.save(csv_path)
                vitals_result = classify_patient_condition(csv_path)

                df = pd.read_csv(csv_path)
                df.rename(columns=lambda x: x.strip().lower(), inplace=True)
                if 'heart rate (bpm)' in df.columns:
                    df.rename(columns={'heart rate (bpm)': 'heart_rate'}, inplace=True)
                if 'blood oxygen level (%)' in df.columns:
                    df.rename(columns={'blood oxygen level (%)': 'spo2'}, inplace=True)
                df = df[['heart_rate', 'spo2']].dropna()
                graph_data = df.to_dict(orient='records')

        if cxr_result and vitals_result:
            conn = get_pg_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO diagnosis_results (user_id, disease, vital_trend)
                VALUES (%s, %s, %s)
            """, (user_id, cxr_result, vitals_result))
            conn.commit()
            cursor.close()
            conn.close()

            latest_report = (cxr_result, vitals_result, "Just now")
            report_message = ""

    return render_template(
        "user_home.html",
        prediction=cxr_result,
        vitals=vitals_result,
        image_path=image_filename,
        graph=graph_data,
        name=name,
        latest_report=latest_report,
        report_message=report_message
    )

# ------------------ ADMIN DASHBOARD ------------------
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect('/login_admin')

    conn = get_pg_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, name, email, mobile FROM users WHERE user_type='user'")
    users = cursor.fetchall()

    cursor.execute("""
        SELECT dr.user_id, u.name, dr.disease, dr.vital_trend, dr.uploaded_at
        FROM diagnosis_results dr
        JOIN users u ON dr.user_id = u.id
        ORDER BY dr.uploaded_at DESC
    """)
    reports = cursor.fetchall()
    
    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html', users=users, reports=reports)

# ------------------ LOGOUT ------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
