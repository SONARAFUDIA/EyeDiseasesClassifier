import argparse
import os
import json
import sys
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

# --- Konfigurasi Environment ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
matplotlib.use('Agg') # Agar tidak muncul window pop-up

DEFAULT_IMG_SIZE = (224, 224)
SEED = 42

# Set seed
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# 1. DEFINISI FOCAL LOSS (Sesuai Notebook)
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
# 2. CALLBACK LOGGER KE JAVA
# ==========================================================
class RealtimeLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Format JSON standar yang dibaca Java
            log_data = {
                "epoch": epoch + 1, 
                "val_loss": float(logs.get('val_loss', 0.0)),
                "val_accuracy": float(logs.get('val_accuracy', 0.0)),
                "loss": float(logs.get('loss', 0.0)),
                "accuracy": float(logs.get('accuracy', 0.0))
            }
            print(json.dumps(log_data))
            sys.stdout.flush()

# ==========================================================
# 3. ARGUMENT PARSER
# ==========================================================
def setup_arg_parser():
    parser = argparse.ArgumentParser()
    
    # Argumen Wajib dari Java
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # Argumen Opsional (Hyperparameters User)
    parser.add_argument("--split-ratio", type=str, default="80,10,10")
    parser.add_argument("--dense-neurons", type=str, default="512")
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    
    # Argumen Load Model (Fitur Lanjut Training)
    parser.add_argument("--load-model-path", type=str, default=None, help="Path file .h5 untuk continue training")

    # Argumen Legacy (Diterima biar Java tidak error, tapi logic di-override)
    parser.add_argument("--freeze-base", action='store_true') 
    parser.add_argument("--balance-data", action='store_true') 

    return parser.parse_args()

# ==========================================================
# 4. PREPARE DATA
# ==========================================================
def prepare_data(args):
    print("--- Mempersiapkan Data ---")
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
    
    return split_dir

# ==========================================================
# 5. BUILD MODEL (STRATEGI NOTEBOOK)
# ==========================================================
def build_or_load_model(num_classes, args):
    # KASUS A: Continue Training (User load model .h5)
    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"--- Memuat Model Lama: {args.load_model_path} ---")
        try:
            # Load model dengan custom object Focal Loss
            model = load_model(args.load_model_path, custom_objects={'focal_loss_fixed': focal_loss()})
            print("Model berhasil dimuat.")
        except Exception as e:
            print(f"Gagal memuat model (mencoba tanpa focal loss): {e}")
            model = load_model(args.load_model_path) # Fallback

        # PENTING: Compile ulang dengan strategi Notebook (LR Kecil + Focal Loss)
        # Ini memastikan model lama 'beradaptasi' dengan teknik training baru
        print("Re-compiling model dengan strategi Notebook (Adam 1e-5, Focal Loss)...")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5), 
            loss=focal_loss(), 
            metrics=['accuracy']
        )
        return model

    # KASUS B: Training Baru (VGG16 ImageNet)
    print("--- Membangun Model VGG16 Baru ---")
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=DEFAULT_IMG_SIZE + (3,))
    
    # Strategi Unfreeze: Block 5 & 4
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name.startswith('block5_conv1') or layer.name.startswith('block4_conv1'):
            set_trainable = True
        layer.trainable = set_trainable
            
    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    neurons_list = [int(n) for n in args.dense_neurons.split(',') if n.strip().isdigit()]
    if not neurons_list: neurons_list = [512]
    
    for neurons in neurons_list:
        x = Dense(neurons, activation='relu')(x)
        if args.dropout_rate > 0:
            x = Dropout(args.dropout_rate)(x)
            
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), 
        loss=focal_loss(), 
        metrics=['accuracy']
    )
    
    return model

# ==========================================================
# 6. MAIN PROCESS
# ==========================================================
def main():
    if sys.stdout.encoding != 'utf-8':
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass

    args = setup_arg_parser()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    # 1. Siapkan Data
    data_dir = prepare_data(args)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # 2. Augmentasi Data (Sesuai Notebook)
    print("--- Menyiapkan Augmentasi Data ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical'
    )
    val_gen = test_datagen.flow_from_directory(
        val_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )

    # 3. Class Weights
    class_weights = None
    try:
        from sklearn.utils import class_weight
        labels = train_gen.classes
        unique_cls = np.unique(labels)
        cw = class_weight.compute_class_weight('balanced', classes=unique_cls, y=labels)
        class_weights = dict(enumerate(cw))
        print(f"Class Weights diaktifkan: {class_weights}")
    except: pass

    # 4. Build / Load Model
    num_classes = len(train_gen.class_indices)
    model = build_or_load_model(num_classes, args)

    # 5. Callbacks
    best_model_path = os.path.join(args.output_dir, "best_model.h5")
    callbacks = [
        RealtimeLoggerCallback(), 
        ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=0)
    ]

    # 6. Training
    print(f"--- Mulai Training ({args.epochs} Epochs) ---")
    try:
        model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0 
        )
    except Exception as e:
        print(f"CRITICAL ERROR saat Training: {e}")
        sys.exit(1)

    # 7. Evaluasi & Simpan Hasil
    print("--- Evaluasi Model ---")
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=DEFAULT_IMG_SIZE, batch_size=args.batch_size, class_mode='categorical', shuffle=False
    )
    
    try:
        model = load_model(best_model_path, custom_objects={'focal_loss_fixed': focal_loss()})
    except: pass

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Save Metrics JSON
    metrics_data = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics_data, f)
        
    # Save Report & Plot
    with open(os.path.join(args.output_dir, "classification_report_test.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix_test.png"))
    
    # Save Sample Predictions for Gallery
    predictions_list = []
    filenames = test_gen.filenames
    labels_map = {v: k for k, v in test_gen.class_indices.items()}
    limit = min(len(filenames), 50)
    for i in range(limit):
        fname = os.path.basename(filenames[i])
        act = labels_map[y_true[i]]
        pred = labels_map[y_pred[i]]
        all_scores = {labels_map[idx]: float(prob) * 100 for idx, prob in enumerate(y_pred_probs[i])}
        predictions_list.append({
            "fileName": fname, "actualLabel": act, "predictedLabel": pred, "confidence": float(np.max(y_pred_probs[i])), "allScores": all_scores
        })
        
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(predictions_list, f)

    print("--- Proses Selesai ---")

if __name__ == "__main__":
    main()