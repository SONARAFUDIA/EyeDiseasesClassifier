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
import splitfolders

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# === IMPORT CALLBACKS ===
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Konfigurasi Awal ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32

# ==========================================================
# 1. CALLBACK UNTUK MONITORING REAL-TIME (JAVA)
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
    parser = argparse.ArgumentParser(description="Backend Training & Evaluasi")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-ratio", type=str, default="70,15,15")
    parser.add_argument("--balance-data", action='store_true')
    parser.add_argument("--freeze-base", action='store_true')
    parser.add_argument("--dense-neurons", type=str, default="512")
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()

# ==========================================================
# 3. FUNGSI PERSIAPAN DATA
# ==========================================================
def prepare_data(args):
    print("--- 1. Mempersiapkan Data ---")
    img_size = DEFAULT_IMG_SIZE
    
    temp_split_dir = os.path.join(args.output_dir, "temp_split_data")
    
    # Hapus temp dir lama jika ada agar bersih
    if os.path.exists(temp_split_dir):
        try:
            shutil.rmtree(temp_split_dir)
        except:
            pass
        
    try:
        ratios = [int(r) / 100.0 for r in args.split_ratio.split(',')]
        print(f"Membagi data ke {temp_split_dir}...")
        splitfolders.ratio(
            args.input_dir, output=temp_split_dir, seed=1337,
            ratio=tuple(ratios), group_prefix=None
        )
    except Exception as e:
        print(f"ERROR Splitfolders: {e}")
        return None, None, None, None

    train_dir = os.path.join(temp_split_dir, 'train')
    val_dir = os.path.join(temp_split_dir, 'val')
    test_dir = os.path.join(temp_split_dir, 'test')

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
        train_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )
    
    return train_generator, val_generator, test_generator, list(train_generator.class_indices.keys())

# ==========================================================
# 4. FUNGSI BANGUN MODEL BARU
# ==========================================================
def build_new_model(args, num_classes):
    # === LOG KHUSUS MODEL BARU ===
    print("\n" + "="*50)
    print("   MODE: MEMBUAT MODEL BARU (NEW TRAINING)")
    print(f"   Base Model   : VGG16 (ImageNet)")
    print(f"   Freeze Base  : {args.freeze_base}")
    print(f"   Dense Neurons: {args.dense_neurons}")
    print(f"   Dropout Rate : {args.dropout_rate}")
    print("="*50 + "\n")
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=DEFAULT_IMG_SIZE + (3,))
    base_model.trainable = not args.freeze_base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    try:
        for neurons in [int(n) for n in args.dense_neurons.split(',') if n.strip()]:
            x = Dense(neurons, activation='relu')(x)
    except:
        x = Dense(512, activation='relu')(x)

    x = Dropout(args.dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)

# ==========================================================
# 5. MAIN FUNCTION
# ==========================================================
def main():
    # Fix Encoding Windows agar tidak crash saat print karakter aneh
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    args = setup_arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Data
    train_gen, val_gen, test_gen, class_names = prepare_data(args)
    if not train_gen: return

    # 2. Model Strategy (Load vs New)
    if args.load_model_path:
        # === LOG KHUSUS LANJUT TRAINING ===
        print("\n" + "="*50)
        print("   MODE: MELANJUTKAN TRAINING (CONTINUE)")
        print(f"   Load File    : {os.path.basename(args.load_model_path)}")
        print("="*50 + "\n")
        
        try:
            model = load_model(args.load_model_path)
            print("Model berhasil dimuat dari disk.")
        except Exception as e:
            print(f"ERROR CRITICAL: Gagal memuat model: {e}")
            return
    else:
        # Buat Model Baru
        model = build_new_model(args, len(class_names))

    # Kompilasi Ulang (Penting untuk reset optimizer state agar fresh)
    print("MENGKOMPILASI MODEL (Adam, lr=0.0001)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', metrics=['accuracy']
    )

    # 3. Training dengan Callbacks Cerdas
    print("--- 3. Memulai Training ---")
    
    best_model_path = os.path.join(args.output_dir, "best_model.h5")
    
    callbacks = [
        RealtimeLoggerCallback(), # Wajib untuk grafik Java
        
        # Simpan hanya yang terbaik
        ModelCheckpoint(best_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        
        # Turunkan LR jika stuck (Kunci agar akurasi naik tinggi)
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        
        # Berhenti jika tidak ada harapan
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    class_weights = None
    if args.balance_data:
        try:
            weights = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
            class_weights = dict(enumerate(weights))
            print("INFO: Class Weights diaktifkan untuk menyeimbangkan data.")
        except: pass

    model.fit(
        train_gen, epochs=args.epochs, validation_data=val_gen,
        callbacks=callbacks, class_weight=class_weights, verbose=2
    )

    # 4. Evaluasi (Gunakan Model Terbaik yang disimpan Checkpoint)
    print(f"Memuat model terbaik dari {best_model_path} untuk evaluasi final...")
    try:
        final_model = load_model(best_model_path)
    except:
        print("Gagal memuat best model, menggunakan model terakhir.")
        final_model = model

    print("--- 4. Evaluasi Final ---")
    y_prob = final_model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # Simpan Metrik
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), 'w') as f:
        json.dump(metrics, f)

    # Simpan Report Teks
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n" + report_text) # Print juga ke log agar user lihat
    with open(os.path.join(args.output_dir, "classification_report_test.txt"), 'w') as f:
        f.write(report_text)

    # Simpan Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix_test.png"))
        plt.close()
    except Exception as e:
        print(f"Warning: Gagal membuat gambar confusion matrix: {e}")

    # Simpan Prediksi untuk Galeri
    preds = []
    filenames = test_gen.filenames
    for i in range(len(filenames)):
        # Ambil top 3 skor biar file tidak terlalu besar
        scores = {class_names[j]: float(y_prob[i][j] * 100) for j in range(len(class_names))}
        
        preds.append({
            "fileName": os.path.basename(filenames[i]),
            "actualLabel": class_names[y_true[i]],
            "predictedLabel": class_names[y_pred[i]],
            "confidence": float(np.max(y_prob[i]) * 100),
            "allScores": scores
        })
    with open(os.path.join(args.output_dir, "predictions.json"), 'w') as f:
        json.dump(preds, f)

    # Rename model akhir untuk konsistensi
    final_save_path = os.path.join(args.output_dir, "trained_model.h5")
    try:
        shutil.move(best_model_path, final_save_path)
        print(f"Model final disimpan di: {final_save_path}")
    except:
        pass
    
    # Bersihkan folder temp
    try: shutil.rmtree(temp_split_dir) 
    except: pass
    
    print("\n--- Proses Selesai ---")

if __name__ == "__main__":
    main()