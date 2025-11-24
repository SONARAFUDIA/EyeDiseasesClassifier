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

# Set backend dan konfigurasi
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Konfigurasi Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
matplotlib.use('Agg') 

DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# 1. CALLBACK: JSON LOGGER (VERSI BERSIH & RAPI)
# ==========================================================
class RealtimeLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Print JSON ini akan muncul SEBELUM ringkasan Keras (karena verbose=2)
            log_data = {
                "epoch": epoch,
                "val_loss": logs.get('val_loss'),
                "val_accuracy": logs.get('val_accuracy')
            }
            print(json.dumps(log_data))
            sys.stdout.flush()

# ==========================================================
# 2. FOCAL LOSS
# ==========================================================
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = K.sum(weight * cross_entropy, axis=1)
        return K.mean(loss)
    return focal_loss_fixed

# ==========================================================
# 3. ARGUMENT PARSER
# ==========================================================
def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-ratio", type=str, default="80,10,10")
    parser.add_argument("--dense-neurons", type=str, default="512")
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-model-path", type=str, default=None)
    parser.add_argument("--freeze-base", action='store_true') # KUNCI: Menentukan mode LR
    parser.add_argument("--balance-data", action='store_true') 
    return parser.parse_args()

# ==========================================================
# 4. PREPARE DATA
# ==========================================================
def prepare_data(args):
    print("--- 1. Mempersiapkan Data ---")
    split_dir = os.path.join(args.output_dir, "split_data")
    
    if not os.path.exists(split_dir) or not os.listdir(split_dir):
        print(f"Melakukan split data dari: {args.input_dir}")
        try:
            ratios = args.split_ratio.split(',')
            if len(ratios) == 3:
                r = [float(i)/100.0 if float(i) > 1.0 else float(i) for i in ratios]
                total = sum(r)
                r = (r[0]/total, r[1]/total, r[2]/total)
            else:
                r = (0.8, 0.1, 0.1)

            splitfolders.ratio(args.input_dir, output=split_dir, seed=SEED, ratio=r, group_prefix=None, move=False)
        except Exception as e:
            print(f"Gagal split data: {e}")
            sys.exit(1)
            
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")
    
    # Augmentasi
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2, 
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
        horizontal_flip=True, fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical'
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )
    
    return train_gen, val_gen, test_gen, list(train_gen.class_indices.keys())

# ==========================================================
# 5. BUILD OR LOAD MODEL (MODIFIKASI CERDAS)
# ==========================================================
def build_or_load_model(num_classes, args):
    # --- KASUS A: Continue Training (User load model .h5) ---
    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"--- Memuat Model Lama: {args.load_model_path} ---")
        try:
            model = load_model(args.load_model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            print("Model berhasil dimuat.")
        except Exception as e:
            print(f"Gagal memuat model (mencoba tanpa focal loss): {e}")
            model = load_model(args.load_model_path)

        # Re-compile selalu pakai strategi Fine Tuning (LR Kecil) karena model sudah jadi
        print("Re-compiling model lama dengan strategi Fine Tuning (Adam 1e-5, Focal Loss)...")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5), 
            loss=focal_loss(), 
            metrics=['accuracy']
        )
        return model

    # --- KASUS B: Training Baru ---
    print("--- Membangun Model VGG16 Baru ---")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=DEFAULT_IMG_SIZE + (3,))
    
    # === LOGIKA MODIFIKASI: CEK FREEZE UNTUK TENTUKAN LR ===
    final_lr = 1e-5 # Default
    
    if args.freeze_base:
        # FASE 1: Freeze Base (Transfer Learning Biasa)
        print("[INFO] Freeze Base AKTIF. Menggunakan Learning Rate Besar (1e-4).")
        base_model.trainable = False
        final_lr = 1e-4 # <-- LR BESAR
    else:
        # FASE 2: Unfreeze Block 4 & 5 (Fine Tuning)
        print("[INFO] Fine Tuning AKTIF. Menggunakan Learning Rate Kecil (1e-5).")
        base_model.trainable = True
        set_trainable = False
        for layer in base_model.layers:
            if layer.name.startswith('block5_conv1') or layer.name.startswith('block4_conv1'):
                set_trainable = True
            layer.trainable = set_trainable
        final_lr = 1e-5 # <-- LR KECIL (Biar model gak rusak)
    # =======================================================
            
    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    try:
        neuron_list = [int(n.strip()) for n in args.dense_neurons.split(',') if n.strip()]
    except:
        neuron_list = [512]
        
    for neurons in neuron_list:
        x = Dense(neurons, activation='relu')(x)
        if args.dropout_rate > 0:
            x = Dropout(args.dropout_rate)(x)
            
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile dengan LR yang sudah ditentukan di atas
    model.compile(
        optimizer=optimizers.Adam(learning_rate=final_lr), 
        loss=focal_loss(), 
        metrics=['accuracy']
    )
    
    return model

# ==========================================================
# 6. EVALUATE & SAVE (CONFUSION MATRIX RAPI)
# ==========================================================
def evaluate_and_save_results(model, test_generator, class_names, output_dir):
    print("--- 4. Mengevaluasi Model ---")
    
    y_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_generator.classes
    filenames = test_generator.filenames

    # Metrik JSON
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    with open(os.path.join(output_dir, "final_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    # --- CONFUSION MATRIX RAPI ---
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
    # -----------------------------

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(output_dir, "classification_report_test.txt"), 'w') as f:
        f.write(report)
        
    # Galeri Prediksi
    predictions_list = []
    for i in range(len(filenames)):
        try:
            fname = os.path.basename(filenames[i])
            act = class_names[y_true[i]]
            pred = class_names[y_pred[i]]
            conf = float(np.max(y_prob[i]) * 100)
            all_scores = {class_names[j]: float(y_prob[i][j] * 100) for j in range(len(class_names))}
            
            predictions_list.append({
                "fileName": fname, "actualLabel": act, "predictedLabel": pred, 
                "confidence": conf, "allScores": all_scores
            })
        except: pass
    
    with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions_list, f, indent=4)

# ==========================================================
# 7. MAIN PROCESS
# ==========================================================
def main():
    if sys.stdout.encoding != 'utf-8':
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass

    args = setup_arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Data
    train_gen, val_gen, test_gen, class_names = prepare_data(args)
    if train_gen is None: return 
    
    # 2. Build Model (Panggil fungsi modifikasi)
    model = build_or_load_model(len(class_names), args)

    # 3. Training
    print("--- 3. Memulai Training ---")
    
    class_weights = None
    if args.balance_data:
        try:
            weights = compute_class_weight(
                class_weight='balanced', classes=np.unique(train_gen.classes), y=train_gen.classes
            )
            class_weights = dict(zip(np.unique(train_gen.classes), weights))
        except: pass

    best_model_path = os.path.join(args.output_dir, "best_model.h5")
    
    callbacks_list = [
        RealtimeLoggerCallback(), 
        ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
    ]

    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=2 # Output ringkas & bersih
    )
    print("Training selesai.")
    
    # 4. Evaluasi
    try: final_model = load_model(best_model_path, custom_objects={'focal_loss_fixed': focal_loss()})
    except: final_model = model

    evaluate_and_save_results(final_model, test_gen, class_names, args.output_dir)
    
    # Cleanup
    try: shutil.move(best_model_path, os.path.join(args.output_dir, "trained_model.h5"))
    except: pass
    try: shutil.rmtree(os.path.join(args.output_dir, "temp_split_data"))
    except: pass

    print("\n--- Proses Selesai ---")

if __name__ == "__main__":
    main()