from model.predictor import DiseasePredictor

predictor = DiseasePredictor(
    dataset_path="dataset/dataset.csv",
    desc_path="dataset/symptom_Description.csv",
    precaution_path="dataset/symptom_precaution.csv",
    symptom_path="dataset/Symptom-severity.csv"
)

if __name__ == "__main__":
    print("Masukkan keluhan/gejala (dalam Bahasa Indonesia):")
    kalimat = input("> ").strip()

    gejala = predictor.extract_symptoms(kalimat)

    if not gejala:
        print("Tidak ada gejala dikenali dari kalimat Anda.")
    else:
        try:
            all_gejala, gejala_valid, gejala_invalid, prediksi = predictor.predict(gejala)

            print("\nGejala dikenali:")
            for g in gejala_valid:
                print(f" - {g.replace('_', ' ')}")

            if gejala_invalid:
                print("\nGejala tidak dikenali:")
                for g in gejala_invalid:
                    print(f" - {g.replace('_', ' ')}")

            print("\nHasil Prediksi:")
            for i, penyakit in enumerate(prediksi, 1):
                print(f"\n{i}. Penyakit: {penyakit['penyakit']} ({penyakit['probabilitas']}%)")
                print(f"   Deskripsi : {penyakit['deskripsi']}")
                for j, t in enumerate(penyakit["tindakan"], 1):
                    print(f"     {j}. {t}")
        except Exception as e:
            print(f"Error: {e}")
