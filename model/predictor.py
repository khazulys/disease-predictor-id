import pandas as pd
import numpy as np
import os
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from deep_translator import GoogleTranslator
from .symptom_map_util import generate_symptom_map
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class DiseasePredictor:
    def __init__(self, dataset_path, desc_path, precaution_path, symptom_path, symptom_map_path="symptom_map.json"):
        self.dataset = pd.read_csv(dataset_path)
        self.desc_df = pd.read_csv(desc_path)
        self.precaution_df = pd.read_csv(precaution_path)
        self.symptom_df = pd.read_csv(symptom_path)

        # Load symptom_map dari file JSON
        if os.path.exists(symptom_map_path):
            with open(symptom_map_path, "r", encoding="utf-8") as f:
                self.symptom_map = json.load(f)
        else:
            self.symptom_map = generate_symptom_map(self.symptom_df["Symptom"])
            with open(symptom_map_path, "w", encoding="utf-8") as f:
                json.dump(self.symptom_map, f, ensure_ascii=False, indent=2)

        # Lanjutkan pelatihan model
        symptom_cols = [col for col in self.dataset.columns if "Symptom" in col]
        self.dataset["Symptoms"] = self.dataset[symptom_cols].values.tolist()
        self.dataset["Symptoms"] = self.dataset["Symptoms"].apply(
            lambda x: [str(i).strip().lower().replace(" ", "_") for i in x if str(i).lower() != "nan"]
        )

        self.mlb = MultiLabelBinarizer()
        X = self.mlb.fit_transform(self.dataset["Symptoms"])
        y = self.dataset["Disease"]

        self.model = MultinomialNB()
        self.model.fit(X, y)
        

    def extract_symptoms(self, sentence_id):
        kalimat = sentence_id.lower()
    
        # Inisialisasi stemmer Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
    
        # Proses stemming seluruh kalimat
        stemmed_sentence = stemmer.stem(kalimat)
        stemmed_words = stemmed_sentence.split()
    
        gejala_ditemukan = []
    
        for indo_symptom, en_symptom in self.symptom_map.items():
            # Stemming gejala
            symptom_stemmed = stemmer.stem(indo_symptom.lower())
            symptom_words = symptom_stemmed.split()
    
            # Cek apakah semua kata dalam symptom muncul di kalimat
            if all(word in stemmed_words for word in symptom_words):
                gejala_ditemukan.append(en_symptom)
    
        return list(set(gejala_ditemukan))

#   def _prepare_data(self):
#         symptom_cols = [col for col in self.df.columns if "Symptom" in col]
#         self.df["Symptoms"] = self.df[symptom_cols].values.tolist()
#         self.df["Symptoms"] = self.df["Symptoms"].apply(
#             lambda x: [str(i).strip().lower().replace(" ", "_") for i in x if str(i).lower() != "nan"]
#         )
#         self.mlb = MultiLabelBinarizer()
#         self.X = self.mlb.fit_transform(self.df["Symptoms"])
#         self.y = self.df["Disease"]
# 
#     def _train_model(self):
#         self.model = MultinomialNB()
#         self.model.fit(self.X, self.y)

    def _ke_indonesia(self, teks):
        try:
            return GoogleTranslator(source='en', target='id').translate(teks)
        except Exception:
            return teks

    def predict(self, symptoms):
        valid = [g for g in symptoms if g in self.mlb.classes_]
        invalid = [g for g in symptoms if g not in self.mlb.classes_]
        if not valid:
            raise ValueError("Tidak ada gejala yang valid ditemukan.")

        x_input = self.mlb.transform([valid])
        probs = self.model.predict_proba(x_input)[0]
        top_idx = np.argsort(probs)[::-1][:3]

        hasil = []
        for idx in top_idx:
            penyakit = self.model.classes_[idx]
            prob = round(probs[idx] * 100, 2)
            deskripsi = self.desc_df[self.desc_df["Disease"] == penyakit]["Description"].values
            tindakan = self.precaution_df[self.precaution_df["Disease"] == penyakit].iloc[:, 1:].values
            hasil.append({
                "penyakit": self._ke_indonesia(penyakit),
                "probabilitas": prob,
                "deskripsi": self._ke_indonesia(deskripsi[0]) if len(deskripsi) else "-",
                "tindakan": [self._ke_indonesia(t) for t in tindakan[0]] if len(tindakan) else []
            })
        return symptoms, valid, invalid, hasil