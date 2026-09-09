from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import model as model_utils
 
app = Flask(__name__)
 
# מאפשר לדף ה-HTML (שיכול לרוץ מכתובת/פורט אחרים, או כקובץ מקומי)
# לבצע בקשות fetch לשרת הזה בלי שהדפדפן יחסום אותן (CORS)
CORS(app)
 
 
@app.errorhandler(FileNotFoundError)
def handle_model_not_found(error):
    """
    אם קובץ המודל (svm_pipeline_model.pkl) או המטא-נתונים (model_metadata.json)
    עדיין לא נוצרו - לדוגמה אם דף האתר נפתח לפני שהריצו את train_model.py -
    מחזירים הודעה ברורה במקום שגיאת שרת "מכוערת" (500 גנרית). כך דף ה-HTML
    יכול להציג הודעה ידידותית כמו "The model file was not found. Please train
    the model and save it before running the app." (בדיוק כמו בדוגמה בקובץ
    ההנחיות של הפרויקט).
    """
    return jsonify({"error": str(error)}), 503
 
 
@app.route("/", methods=["GET"])
def index():
    """Serve the eligibility interface from the same origin as the API."""
    return send_from_directory(".", "index.html")


@app.route("/<path:asset>", methods=["GET"])
def frontend_asset(asset):
    """Serve the frontend stylesheet and client script."""
    if asset in {"styles.css", "app.js"}:
        return send_from_directory(".", asset)
    return jsonify({"error": "Not found"}), 404
 
 
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
 
 
@app.route("/model/predict", methods=["POST"])
def model_predict():
    """
    שלב 3 - מקבלת מהטופס באתר JSON עם פרטי בקשת הלוואה בודדת, ומחזירה
    את חיזוי המודל (Loan Approved / Loan Not Approved).
 
    גוף הבקשה (JSON) חייב להכיל ערך לכל אחד מהפיצ'רים הבאים:
    CoapplicantIncome, ApplicantIncome, LoanAmount, Credit_History,
    Loan_Amount_Term, Self_Employed, Education
    (אפשר לקבל את הרשימה המדויקת גם מ-GET /model/features)
    """
    request_data = request.get_json(silent=True) or {}
 
    try:
        result = model_utils.predict_loan_status(request_data)
    except ValueError as error:
        # שדות חסרים/לא תקינים בבקשה - שגיאת לקוח (400), לא שגיאת שרת
        return jsonify({"error": str(error)}), 400
 
    return jsonify(result)
 
 
if __name__ == "__main__":
    # debug=True נוח לפיתוח (מציג שגיאות מפורטות, טוען מחדש אוטומטית) -
    # יש לכבות (debug=False) לפני פריסה אמיתית לאינטרנט (למשל ב-Render)
    app.run(debug=True, port=5000)
 