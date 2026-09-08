"""
שלב 2 - פונקציות עזר שחושפות את נתוני המודל.

המודול הזה אחראי אך ורק על טעינת ה-Pipeline המאומן ומטא-הנתונים השמורים
(model_metadata.json, שנוצר על ידי train_model.py), ועל חשיפתם כפונקציות
פייתון "נקיות" - כל אחת מחזירה מבנה נתונים פשוט (dict / list) שקל להפוך
ל-JSON. app.py רק עוטף את הפונקציות האלה ב-endpoints של REST API, ולא
מכיל בעצמו שום לוגיקה של טעינת קבצים או חישוב.
"""

import json
import os

import joblib
import pandas as pd

MODEL_FILE_PATH = "svm_pipeline_model.pkl"
METADATA_FILE_PATH = "model_metadata.json"

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
        if not os.path.exists(METADATA_FILE_PATH):
            raise FileNotFoundError(
                f"קובץ המטא-נתונים '{METADATA_FILE_PATH}' לא נמצא. "
                "יש להריץ קודם את train_model.py כדי ליצור אותו."
            )
        with open(METADATA_FILE_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)


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
    return _metadata["features"]


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