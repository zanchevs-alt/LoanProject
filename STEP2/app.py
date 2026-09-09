"""
שלב 2 - REST API שחושף את נתוני המודל (Flask).

כל endpoint כאן הוא "עטיפה" דקה סביב פונקציה מ-model_utils.py - השרת עצמו
לא מכיל שום לוגיקה של טעינת קבצים או חישוב, רק ממיר את הפלט של הפונקציות
ל-JSON ומחזיר אותו ללקוח (דף ה-HTML שנבנה בהמשך שלב 2, בעזרת vibe coding).

איך מריצים:
    python app.py
השרת יעלה בכתובת: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

import model

app = Flask(__name__)
PROJECT_DIR = __file__.rsplit("\\", 1)[0]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Loan Lens | Model Observatory</title>
<style>
:root{--ink:#17211b;--muted:#68746d;--paper:#f4f1e8;--lime:#c7ed62;--coral:#ff765f;--line:#d8ded0}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,serif}main{max-width:1180px;margin:auto;padding:32px 24px 60px}.top{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:20px;font:600 12px Arial;letter-spacing:2px;text-transform:uppercase}.mark{display:flex;gap:10px}.dot{width:13px;height:13px;background:var(--coral);border-radius:50%}.status{color:#50733b}.hero{display:grid;grid-template-columns:1.25fr .75fr;gap:28px;padding:68px 0 42px}.eyebrow{font:700 12px Arial;letter-spacing:2px;color:var(--coral);text-transform:uppercase}.hero h1{font-size:clamp(48px,8vw,96px);line-height:.9;font-weight:400;margin:14px 0 24px;letter-spacing:-3px}.hero p{font:17px Arial;line-height:1.6;max-width:510px;color:var(--muted)}.stamp{background:var(--ink);color:var(--lime);padding:30px;min-height:220px;display:flex;flex-direction:column;justify-content:space-between}.stamp strong{font:700 64px Arial}.stamp span{font:12px Arial;text-transform:uppercase;letter-spacing:1px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#fffdf8;border:1px solid var(--line);padding:22px;min-height:150px}.card h2{font:700 11px Arial;letter-spacing:1.5px;text-transform:uppercase;margin:0 0 22px;color:var(--muted)}.value{font:42px Arial;font-weight:700}.wide{grid-column:span 2}.features{display:flex;flex-wrap:wrap;gap:8px}.pill{border:1px solid var(--line);padding:8px 10px;font:12px Arial}.samples{margin-top:14px;overflow:auto;background:#fffdf8;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font:12px Arial}th,td{text-align:left;padding:13px 16px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);text-transform:uppercase;font-size:10px;letter-spacing:1px}.badge{background:var(--lime);padding:5px 8px;font-weight:bold}.foot{font:12px Arial;color:var(--muted);margin-top:28px}@media(max-width:720px){main{padding:20px 16px}.hero{grid-template-columns:1fr;padding-top:45px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero h1{letter-spacing:-2px}}
</style></head><body><main><header class="top"><div class="mark"><i class="dot"></i> Loan Lens</div><div class="status">● Live model observatory</div></header>
<section class="hero"><div><div class="eyebrow">SVM / loan approval</div><h1>See the thinking<br>behind the yes.</h1><p>A clear, living view of the model's training story, signals, and confidence.</p></div><div class="stamp"><span>Test accuracy</span><strong id="accuracy">--</strong><span id="modelName">Loading model profile...</span></div></section>
<section class="grid"><article class="card"><h2>Training set</h2><div class="value" id="trainRows">--</div><div class="foot">rows used to learn</div></article><article class="card"><h2>Test set</h2><div class="value" id="testRows">--</div><div class="foot">fresh rows evaluated</div></article><article class="card"><h2>Support vectors</h2><div class="value" id="vectors">--</div><div class="foot">boundary-defining examples</div></article><article class="card wide"><h2>Signals in the model</h2><div class="features" id="features"></div></article><article class="card"><h2>Decision margins</h2><div class="value" id="marginCount">--</div><div class="foot">test confidence readings</div></article></section>
<section class="samples"><table><thead><tr><th>Applicant income</th><th>Loan amount</th><th>Education</th><th>Self employed</th><th>Decision</th></tr></thead><tbody id="samples"></tbody></table></section><div class="foot">Model metadata refreshed from the latest training run.</div></main>
<script>Promise.all(['/model/info','/model/features','/model/metrics','/model/margins','/model/samples'].map(u=>fetch(u).then(r=>r.json()))).then(([info,features,metrics,margins,samples])=>{document.querySelector('#accuracy').textContent=(metrics.accuracy*100).toFixed(1)+'%';document.querySelector('#trainRows').textContent=info.model_info.train_rows;document.querySelector('#testRows').textContent=info.model_info.test_rows;document.querySelector('#vectors').textContent=info.model_info.total_support_vectors;document.querySelector('#marginCount').textContent=margins.length;document.querySelector('#modelName').textContent=info.model_info.algorithm+' · '+info.model_info.kernel+' kernel';document.querySelector('#features').innerHTML=features.all.map(x=>'<span class="pill">'+x+'</span>').join('');document.querySelector('#samples').innerHTML=samples.slice(0,6).map(x=>'<tr><td>'+x.ApplicantIncome+'</td><td>'+x.LoanAmount+'</td><td>'+x.Education+'</td><td>'+x.Self_Employed+'</td><td><span class="badge">'+x.Loan_Status+'</span></td></tr>').join('')}).catch(()=>document.querySelector('#accuracy').textContent='Offline');</script></body></html>"""

# מאפשר לדף ה-HTML (שיכול לרוץ מכתובת/פורט אחרים, או כקובץ מקומי)
# לבצע בקשות fetch לשרת הזה בלי שהדפדפן יחסום אותן (CORS)
CORS(app)


@app.errorhandler(FileNotFoundError)
def model_file_missing(error):
    """Tell API clients how to recover when training artifacts are absent."""
    return jsonify({
        "error": "model_not_found",
        "message": "The model file was not found. Please train the model and save it before running the app.",
    }), 503


@app.errorhandler(503)
def service_unavailable(error):
    """Provide a consistent friendly response while the model is initializing."""
    return jsonify({
        "error": "service_unavailable",
        "message": "The system is initializing, please wait.",
    }), 503


@app.route("/", methods=["GET"])
def index():
    """Serve the dashboard saved in the STEP2 project folder."""
    return send_from_directory(PROJECT_DIR, "index.html")


@app.route("/model/info", methods=["GET"])
def model_info():
    """מחזיר סקירה כללית של המודל: אלגוריתם, פיצ'רים, מחלקות, קובץ המודל."""
    return jsonify(model.get_full_model_overview())


@app.route("/model/features", methods=["GET"])
def model_features():
    """מחזיר את רשימת הפיצ'רים שנבחרו לאימון המודל."""
    return jsonify(model.get_features())


@app.route("/model/samples", methods=["GET"])
def model_samples():
    """מחזיר כמה שורות לדוגמה מתוך הדאטהסט המקורי."""
    return jsonify(model.get_samples())


@app.route("/model/metrics", methods=["GET"])
def model_metrics():
    """מחזיר את מדדי הדיוק של המודל (accuracy / confusion matrix / classification report)."""
    return jsonify(model.get_metrics())


@app.route("/model/margins", methods=["GET"])
def model_margins():
    """מחזיר את התפלגות ערכי ה-margin על קבוצת הבדיקה (לגרף היסטוגרמה)."""
    return jsonify(model.get_margins_distribution())


if __name__ == "__main__":
    # debug=True נוח לפיתוח (מציג שגיאות מפורטות, טוען מחדש אוטומטית) -
    # יש לכבות (debug=False) לפני פריסה אמיתית לאינטרנט (למשל ב-Render)
    app.run(host="127.0.0.1", debug=False, port=5000)