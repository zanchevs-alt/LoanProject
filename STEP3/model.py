import json
import os
 
import joblib
import pandas as pd
 
MODEL_FILE_PATH = "svm_pipeline_model.pkl"
METADATA_FILE_PATH = "model_metadata.json"
DEFAULT_FEATURES = [
    "CoapplicantIncome",
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History",
    "Loan_Amount_Term",
    "Self_Employed",
    "Education",
]
 
# משתנים "פרטיים" למודול - נטענים פעם אחת בלבד (בזמן ה-import הראשון),
# כדי שלא נצטרך לטעון מהדיסק בכל קריאה ל-API
_pipeline = None
_metadata = None
 
 
def _ensure_loaded() -> None:
    """טוען את המודל ואת מטא-הנתונים לזיכרון, פעם אחת בלבד (lazy load)."""
    global _pipeline, _metadata
 
    if _pipeline is None:
        if not os.path.exists(MODEL_FILE_PATH):
            raise FileNotFoundError(
                f"קובץ המודל '{MODEL_FILE_PATH}' לא נמצא. "
                "יש להריץ קודם את train_model.py כדי לאמן ולשמור את המודל."
            )
        _pipeline = joblib.load(MODEL_FILE_PATH)
 
    if _metadata is None:
        if os.path.exists(METADATA_FILE_PATH):
            with open(METADATA_FILE_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            _metadata = {}
 
 
def get_pipeline():
    """מחזיר את ה-Pipeline המאומן (StandardScaler/OneHotEncoder + SVC)."""
    _ensure_loaded()
    return _pipeline
 
 
def get_model_info() -> dict:
    """פרטי המודל עצמו: אלגוריתם, kernel, מספר Support Vectors וכו'."""
    _ensure_loaded()
    return _metadata["model_info"]
 
 
def get_features() -> dict:
    """רשימת הפיצ'רים שנבחרו לאימון, מחולקים לפי סוג הטיפול שקיבלו."""
    _ensure_loaded()
    return _metadata.get("features", {"all": DEFAULT_FEATURES})
 
 
def get_samples() -> list:
    """כמה שורות לדוגמה מתוך הדאטהסט המקורי, להצגה בדף."""
    _ensure_loaded()
    return _metadata["samples"]
 
 
def get_metrics() -> dict:
    """מדדי הדיוק של המודל: accuracy, confusion matrix, classification report."""
    _ensure_loaded()
    return _metadata["metrics"]
 
 
def get_margins_distribution() -> list:
    """ערכי ה-margin (decision_function) על קבוצת הבדיקה - להיסטוגרמה."""
    _ensure_loaded()
    return _metadata["margins_on_test_set"]
 
 
def get_model_file_info() -> dict:
    """שם קובץ המודל השמור וגודלו."""
    _ensure_loaded()
    return _metadata["model_file"]
 
 
def get_classes() -> list:
    """התוויות האפשריות שהמודל חוזה (למשל ['N', 'Y'])."""
    _ensure_loaded()
    return _metadata["classes"]
 
 
def compute_margin(input_df: pd.DataFrame):
    """
    מחשבת את ה-margin (decision_function) עבור שורת/שורות קלט חדשות.
 
    ה-margin הוא המרחק המסומן (signed distance) של הדוגמה ממישור המפריד
    (hyperplane) של ה-SVM: ערך חיובי משמעו שהדוגמה בצד של המחלקה החיובית
    (Y), ערך שלילי - בצד המחלקה השנייה (N), וככל שהערך רחוק יותר מ-0 -
    המודל "בטוח" יותר בחיזוי. הפונקציה הזו תשמש גם את ה-predict בשלב 3.
 
    input_df: DataFrame עם אותן עמודות הפיצ'רים המקוריות (לא מקודדות) -
    הקידוד והנרמול קורים אוטומטית בתוך ה-Pipeline.
    """
    pipeline = get_pipeline()
    return pipeline.decision_function(input_df).tolist()
 
 
def get_full_model_overview() -> dict:
    """מרכזת את כל המידע על המודל למקום אחד - נוח ל-endpoint אחד מסכם."""
    return {
        "model_info": get_model_info(),
        "features": get_features(),
        "classes": get_classes(),
        "model_file": get_model_file_info(),
    }
 
 
# --- שלב 3: פונקציית החיזוי עבור בקשה בודדת מהמשתמש ---
 
APPROVED_CLASS = "Y"  # הערך ב-Loan_Status שמסמן הלוואה מאושרת
 
 
def predict_loan_status(request_data: dict) -> dict:
    """
    מקבלת dict אחד עם פרטי בקשת הלוואה (כמו שמגיע מהטופס באתר), ומחזירה
    את חיזוי המודל: התוצאה הגולמית (Y/N), האם ההלוואה אושרה, ה-margin
    (מידת ה"ביטחון" של המודל), והודעה מוכנה להצגה למשתמש.
 
    request_data חייב להכיל ערך לכל אחד מהפיצ'רים שהמודל אומן עליהם
    (הרשימה נמצאת ב-get_features()["all"]) - אחרת נזרקת ValueError עם
    רשימת השדות החסרים, כדי שה-API יוכל להחזיר שגיאה ברורה ללקוח.
    """
    required_features = get_features()["all"]
 
    missing = [f for f in required_features if f not in request_data]
    if missing:
        raise ValueError(f"חסרים השדות הבאים בבקשה: {', '.join(missing)}")
 
    # בונים DataFrame של שורה אחת, בדיוק באותן עמודות שהמודל מצפה להן
    input_row = pd.DataFrame([{f: request_data[f] for f in required_features}])
 
    pipeline = get_pipeline()
    prediction = pipeline.predict(input_row)[0]
    margin = float(pipeline.decision_function(input_row)[0])
 
    approved = bool(prediction == APPROVED_CLASS)
 
    return {
        "prediction": str(prediction),
        "approved": approved,
        "margin": round(margin, 4),
        "message": "Loan Approved" if approved else "Loan Not Approved",
    }
 