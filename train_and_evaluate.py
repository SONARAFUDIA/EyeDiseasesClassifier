import argparse
import os
import json
import sys
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
import splitfolders # <-- Pastikan ini terinstall (pip install split-folders)

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Konfigurasi Awal ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
matplotlib.use('Agg') # Mode non-GUI untuk matplotlib

DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32

# ==========================================================
# 1. CALLBACK UNTUK MONITORING REAL-TIME
# ==========================================================
class RealtimeLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            log_data = {
                "epoch": epoch,
                "val_loss": logs.get('val_loss'),
                "val_accuracy": logs.get('val_accuracy')
            }
            print(json.dumps(log_data))
            sys.stdout.flush()

# ==========================================================
# 2. FUNGSI ARGUMENT PARSER
# ==========================================================
def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Backend Training & Evaluasi Model Klasifikasi Gambar")
    
    # --- Argumen Modul 1: Data ---
    parser.add_argument("--input-dir", type=str, required=True, help="Path ke folder dataset input (Ground Truth).")
    parser.add_argument("--output-dir", type=str, required=True, help="Path ke folder untuk menyimpan hasil.")
    parser.add_argument("--split-ratio", type=str, default="70,15,15", help="Rasio Train,Val,Test dipisah koma (e.g., '70,15,15')")
    parser.add_argument("--balance-data", action='store_true', help="Terapkan class weights untuk data imbalanced.")
    
    # --- Argumen Modul 2: Model & Training ---
    parser.add_argument("--freeze-base", action='store_true', help="Bekukan (freeze) bobot pre-trained VGG16.")
    parser.add_argument("--dense-neurons", type=str, default="512", help="Jumlah neuron di layer Dense (pisahkan koma, e.g., '512,256')")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Rate untuk layer Dropout (e.g., 0.5)")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch training.")
    
    # --- Argumen Lain ---
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Ukuran batch.")
    
    return parser.parse_args()

# ==========================================================
# 3. FUNGSI PERSIAPAN DATA
# ==========================================================
def prepare_data(args):
    print("--- 1. Mempersiapkan Data ---")
    img_size = DEFAULT_IMG_SIZE
    
    # --- 1a. Membagi Folder ---
    temp_split_dir = os.path.join(args.output_dir, "temp_split_data")
    if os.path.exists(temp_split_dir):
        print(f"Menghapus folder split sementara sebelumnya di: {temp_split_dir}")
        shutil.rmtree(temp_split_dir)
        
    try:
        ratios = [int(r) / 100.0 for r in args.split_ratio.split(',')]
        if len(ratios) != 3: raise ValueError("Split ratio harus 3 angka")
        
        print(f"Membagi data ke {temp_split_dir} dengan rasio (Train/Val/Test): {ratios}")
        splitfolders.ratio(
            args.input_dir,
            output=temp_split_dir,
            seed=1337,
            ratio=tuple(ratios), # (train_ratio, val_ratio, test_ratio)
            group_prefix=None
        )
        print("Pembagian data selesai.")
    except Exception as e:
        print(f"ERROR: Gagal membagi folder dengan 'split-folders': {e}")
        return None, None, None, None

    train_dir = os.path.join(temp_split_dir, 'train')
    val_dir = os.path.join(temp_split_dir, 'val')
    test_dir = os.path.join(temp_split_dir, 'test')

    # --- 1b. Membuat Generator Data ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    print(f"Ditemukan {len(class_names)} kelas: {class_names}")

    return train_generator, val_generator, test_generator, class_names

# ==========================================================
# 4. FUNGSI PEMBUATAN MODEL
# ==========================================================
def build_model(args, num_classes):
    print("--- 2. Membangun Model VGG16 ---")
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=DEFAULT_IMG_SIZE + (3,))

    if args.freeze_base:
        print("Membekukan (freeze) bobot VGG16 base model.")
        base_model.trainable = False
    else:
        print("Bobot VGG16 base model akan di-fine-tune (trainable).")
        base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    try:
        neuron_list = [int(n.strip()) for n in args.dense_neurons.split(',') if n.strip()]
        if not neuron_list: raise ValueError("Tidak ada neuron valid")
        
        print(f"Menambahkan layer Dense kustom dengan neuron: {neuron_list}")
        for neurons in neuron_list:
            x = Dense(neurons, activation='relu')(x)
    except Exception as e:
        print(f"Peringatan: Gagal parse dense-neurons ('{args.dense_neurons}'). Menggunakan default 512. Error: {e}")
        x = Dense(512, activation='relu')(x)

    print(f"Menambahkan layer Dropout dengan rate: {args.dropout_rate}")
    x = Dropout(args.dropout_rate)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model berhasil dibangun dan dikompilasi.")
    model.summary()
    return model

# ==========================================================
# 5. FUNGSI EVALUASI & PENYIMPANAN HASIL
# ==========================================================
def evaluate_and_save_results(model, test_generator, class_names, output_dir):
    print("--- 4. Mengevaluasi Model dengan Test Set ---")
    
    # Dapatkan prediksi
    y_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    # Dapatkan nama file
    filenames = test_generator.filenames # e.g., ['Glaucoma/img1.png', 'Healthy/img2.png']

    # --- 4a. Hitung Metrik ---
    print("--- 4a. Menghitung Metrik Final ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }
    
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrik final disimpan di: {metrics_path}")
    except Exception as e:
        print(f"Gagal menyimpan metrics.json: {e}")

    # --- 4b. Buat & Simpan Confusion Matrix ---
    print("--- 4b. Membuat Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix_test.png")
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Test Set)', fontsize=16)
        plt.ylabel('Label Asli', fontsize=12)
        plt.xlabel('Label Prediksi', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix disimpan di: {cm_path}")
    except Exception as e:
        print(f"Gagal menyimpan confusion_matrix_test.png: {e}")

    # --- 4c. Buat & Simpan Classification Report ---
    print("--- 4c. Membuat Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, "classification_report_test.txt")
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report disimpan di: {report_path}")
        print("\n--- Classification Report (Test Set) ---")
        print(report)
    except Exception as e:
        print(f"Gagal menyimpan classification_report_test.txt: {e}")
        
    # --- 4d. (BARU) Membuat & Menyimpan Daftar Prediksi (Galeri) ---
    print("--- 4d. Membuat Daftar Prediksi Detail (Galeri) ---")
    predictions_list = []
    
    if not (len(y_true) == len(y_pred) == len(filenames) == len(y_prob)):
         print("Peringatan: Ketidakcocokan panjang array, galeri mungkin tidak lengkap.")

    for i in range(len(filenames)):
        try:
            file_path_relative = filenames[i] # 'Glaucoma/img1.png'
            file_name_only = os.path.basename(file_path_relative) # 'img1.png'
            
            actual_label = class_names[y_true[i]]
            predicted_label = class_names[y_pred[i]]
            
            confidence = float(np.max(y_prob[i]) * 100)
            all_scores = {class_names[j]: float(y_prob[i][j] * 100) for j in range(len(class_names))}
            
            predictions_list.append({
                "fileName": file_name_only,
                "actualLabel": actual_label, # The class name, e.g., 'Glaucoma'
                "predictedLabel": predicted_label,
                "confidence": confidence,
                "allScores": all_scores
            })
        except IndexError:
             print(f"Peringatan: Melewati index {i} saat membuat galeri.")
    
    preds_path = os.path.join(output_dir, "predictions.json")
    try:
        with open(preds_path, 'w') as f:
            json.dump(predictions_list, f, indent=4)
        print(f"Hasil prediksi detail (galeri) disimpan di: {preds_path}")
    except Exception as e:
        print(f"Gagal menyimpan predictions.json: {e}")

# ==========================================================
# 6. FUNGSI UTAMA (MAIN)
# ==========================================================
def main():
    args = setup_arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Tahap 1: Persiapan Data ---
    train_gen, val_gen, test_gen, class_names = prepare_data(args)
    if train_gen is None: return
    num_classes = len(class_names)

    # --- Tahap 2: Bangun Model ---
    model = build_model(args, num_classes)
    
    # --- Tahap 3: Training ---
    print("--- 3. Memulai Training ---")
    class_weights = None
    if args.balance_data:
        try:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_gen.classes),
                y=train_gen.classes
            )
            class_weights = dict(zip(np.unique(train_gen.classes), weights))
            print(f"Menerapkan Class Weights untuk balancing: {class_weights}")
        except Exception as e:
            print(f"Peringatan: Gagal menghitung class weights. Error: {e}")
            class_weights = None

    realtime_logger = RealtimeLoggerCallback()
    callbacks_list = [realtime_logger]

    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=2
    )
    print("Training selesai.")
    
    # --- Tahap 4: Evaluasi & Simpan ---
    evaluate_and_save_results(model, test_gen, class_names, args.output_dir)
    
    # --- Tahap 5: Simpan Model ---
    model_save_path = os.path.join(args.output_dir, "trained_model.h5")
    try:
        model.save(model_save_path)
        print(f"Model terlatih berhasil disimpan di: {model_save_path}")
    except Exception as e:
        print(f"Gagal menyimpan model .h5: {e}")
        
    # --- Tahap 6: Bersihkan Folder Temp ---
    temp_split_dir = os.path.join(args.output_dir, "temp_split_data")
    try:
        if os.path.exists(temp_split_dir):
            shutil.rmtree(temp_split_dir)
            print(f"Folder data sementara {temp_split_dir} berhasil dihapus.")
    except Exception as e:
        print(f"Peringatan: Gagal menghapus folder sementara {temp_split_dir}. Error: {e}")

    print("\n--- Proses Selesai ---")

if __name__ == "__main__":
    main()