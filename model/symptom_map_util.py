import json
import os
from deep_translator import GoogleTranslator

def generate_symptom_map(symptom_list, filepath="symptom_map.json"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    mapping = {}
    for symptom in symptom_list:
        symptom_clean = str(symptom).strip().lower().replace("_", " ")
        try:
            indo = GoogleTranslator(source='en', target='id').translate(symptom_clean)
            mapping[indo] = symptom.strip().lower()
        except Exception:
            continue

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    return mapping