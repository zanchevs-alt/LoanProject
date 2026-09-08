"""
שלב 2 - REST API שחושף את נתוני המודל (Flask).

כל endpoint כאן הוא "עטיפה" דקה סביב פונקציה מ-model_utils.py - השרת עצמו
לא מכיל שום לוגיקה של טעינת קבצים או חישוב, רק ממיר את הפלט של הפונקציות
ל-JSON ומחזיר אותו ללקוח (דף ה-HTML שנבנה בהמשך שלב 2, בעזרת vibe coding).

איך מריצים:
    python app.py
השרת יעלה בכתובת: http://127.0.0.1:5000
"""

from flask import Flask, jsonify
from flask_cors import CORS

import model_utils

app = Flask(__name__)

# מאפשר לדף ה-HTML (שיכול לרוץ מכתובת/פורט אחרים, או כקובץ מקומי)
# לבצע בקשות fetch לשרת הזה בלי שהדפדפן יחסום אותן (CORS)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    """דף בדיקה בסיסי - מוודא שהשרת פועל."""
    return jsonify({"status": "ok", "message": "Loan approval model API is running"})


@app.route("/model/info", methods=["GET"])
def model_info():
    """מחזיר סקירה כללית של המודל: אלגוריתם, פיצ'רים, מחלקות, קובץ המודל."""
    return jsonify(model_utils.get_full_model_overview())


@app.route("/model/features", methods=["GET"])
def model_features():
    """מחזיר את רשימת הפיצ'רים שנבחרו לאימון המודל."""
    return jsonify(model_utils.get_features())


@app.route("/model/samples", methods=["GET"])
def model_samples():
    """מחזיר כמה שורות לדוגמה מתוך הדאטהסט המקורי."""
    return jsonify(model_utils.get_samples())


@app.route("/model/metrics", methods=["GET"])
def model_metrics():
    """מחזיר את מדדי הדיוק של המודל (accuracy / confusion matrix / classification report)."""
    return jsonify(model_utils.get_metrics())


@app.route("/model/margins", methods=["GET"])
def model_margins():
    """מחזיר את התפלגות ערכי ה-margin על קבוצת הבדיקה (לגרף היסטוגרמה)."""
    return jsonify(model_utils.get_margins_distribution())


if __name__ == "__main__":
    # debug=True נוח לפיתוח (מציג שגיאות מפורטות, טוען מחדש אוטומטית) -
    # יש לכבות (debug=False) לפני פריסה אמיתית לאינטרנט (למשל ב-Render)
    app.run(debug=True, port=5000)