package eyepred.eyediseasesclassifier;

// Import JavaFX
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;

// Import Java IO & Util
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

// Import Jackson JSON
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;


public class MainController {

    // === Daftar Kelas (HARUS SAMA DENGAN OUTPUT PYTHON) ===
    private static final String[] KELAS = {
            "Central Serous Chorioretinopathy", "Diabetic Retinopathy",
            "Disc Edema", "Glaucoma", "Healthy", "Macular Scar", "Myopia",
            "Pterygium", "Retinal Detachment", "Retinitis Pigmentosa"
    };

    // === Injeksi FXML (Panel Kontrol) ===
    @FXML private Button btnInputDataset;
    @FXML private Label lblInputPath;
    @FXML private Button btnOutput;
    @FXML private Label lblOutputPath;
    @FXML private Button btnModelPath;
    @FXML private Label lblModelPath;
    @FXML private Slider sliderSplit; // Tidak dipakai backend evaluasi
    @FXML private Label lblSplitValue;
    @FXML private TextField txtEpochs; // Tidak dipakai
    @FXML private CheckBox chkTransferLearning; // Tidak dipakai
    @FXML private TextField txtDropout; // Tidak dipakai
    @FXML private TextField txtNeurons; // Tidak dipakai
    @FXML private Button btnMulaiTraining; // Tombol Aksi

    // === Injeksi FXML (Area Output) ===
    @FXML private VBox loadingPane;
    @FXML private Label lblLoadingStatus;
    @FXML private ProgressBar progressBar;
    @FXML private VBox resultsPane;
    @FXML private Label lblAkurasi;
    @FXML private Label lblPresisi;
    @FXML private Label lblRecall;
    @FXML private Label lblF1;
    @FXML private ImageView imgConfusionMatrix;
    @FXML private ListView<GalleryItem> galleryListView;

    // ObjectMapper untuk membaca JSON
    private final ObjectMapper objectMapper = new ObjectMapper();

    // Variabel untuk menyimpan path File objek
    private File inputDir = null;
    private File outputDir = null;
    private File modelFile = null;

    // Model data untuk galeri (sesuaikan field dengan JSON)
    public static class GalleryItem {
        public String fileName;
        public String actualLabel;
        public String predictedLabel;
        public double confidence;
        public Map<String, Double> allScores;
        public GalleryItem() {} // Constructor default
        // Getter bisa ditambahkan jika perlu, atau pakai field public
    }

    @FXML
    public void initialize() {
        // Setup slider (hanya visual)
        sliderSplit.valueProperty().addListener((obs, oldVal, newVal) -> {
            int train = newVal.intValue();
            int test = 100 - train;
            lblSplitValue.setText(String.format("%d%% / %d%%", train, test));
        });

        // Kondisi Awal UI
        loadingPane.setVisible(false);
        loadingPane.setManaged(false); // Agar tidak makan tempat saat disembunyikan
        resultsPane.setVisible(true);
        clearResultsUI(); // Kosongkan tampilan hasil

        setupGalleryClickListener();
        setupGalleryCellFactory(); // Setup tampilan galeri
    }

    // === Handlers Tombol ===

    @FXML
    private void handleInputDataset() {
        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Pilih Folder Dataset Input (Ground Truth)");
        File selectedDir = dirChooser.showDialog(btnInputDataset.getScene().getWindow());
        if (selectedDir != null && selectedDir.isDirectory()) {
            inputDir = selectedDir;
            lblInputPath.setText(inputDir.getAbsolutePath());
        } else {
            inputDir = null;
            lblInputPath.setText("Belum ada folder dipilih");
        }
    }

    @FXML
    private void handleOutputFolder() {
        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Pilih Folder Output Hasil");
        File selectedDir = dirChooser.showDialog(btnOutput.getScene().getWindow());
        if (selectedDir != null && selectedDir.isDirectory()) {
            outputDir = selectedDir;
            lblOutputPath.setText(outputDir.getAbsolutePath());
        } else {
            outputDir = null;
            lblOutputPath.setText("Belum ada folder dipilih");
        }
    }

    @FXML
    private void handleModelPath() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Pilih File Model (.h5)");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Keras Model", "*.h5")
        );
        File selectedFile = fileChooser.showOpenDialog(btnModelPath.getScene().getWindow());
        if (selectedFile != null && selectedFile.exists()) {
            modelFile = selectedFile;
            lblModelPath.setText(modelFile.getName());
        } else {
            modelFile = null;
            lblModelPath.setText("Belum ada model dipilih");
        }
    }

    @FXML
    private void handleMulaiTraining() { // Nama fungsi tetap, aksi jadi evaluasi
        // --- Validasi Input ---
        if (modelFile == null) {
            showAlert(Alert.AlertType.ERROR, "Error Input", "Silakan pilih file model (.h5) yang valid.");
            return;
        }
        if (inputDir == null) {
            showAlert(Alert.AlertType.ERROR, "Error Input", "Silakan pilih folder dataset input yang valid.");
            return;
        }
        if (outputDir == null) {
            showAlert(Alert.AlertType.ERROR, "Error Input", "Silakan pilih folder output yang valid.");
            return;
        }

        // Kosongkan hasil sebelumnya
        clearResultsUI();

        // --- Jalankan Task Python di Background ---
        Task<Boolean> evaluationTask = createPythonEvaluationTask();

        // --- Ikat UI ke Task ---
        progressBar.progressProperty().unbind();
        progressBar.progressProperty().bind(evaluationTask.progressProperty());
        lblLoadingStatus.textProperty().bind(evaluationTask.messageProperty());

        // --- Atur Aksi UI ---
        evaluationTask.setOnRunning(e -> {
            resultsPane.setVisible(false);
            loadingPane.setVisible(true);
            loadingPane.setManaged(true);
            btnMulaiTraining.setDisable(true);
        });

        evaluationTask.setOnSucceeded(e -> {
            loadingPane.setVisible(false);
            loadingPane.setManaged(false);
            resultsPane.setVisible(true);
            btnMulaiTraining.setDisable(false);

            if (evaluationTask.getValue()) { // Jika exitCode == 0
                loadResults(); // Muat hasil dari file output
            } else {
                showAlert(Alert.AlertType.ERROR, "Error Evaluasi", "Proses evaluasi Python gagal. Periksa log konsol.");
            }
        });

        evaluationTask.setOnFailed(e -> {
            loadingPane.setVisible(false);
            loadingPane.setManaged(false);
            resultsPane.setVisible(true);
            btnMulaiTraining.setDisable(false);
            showAlert(Alert.AlertType.ERROR, "Error Eksekusi", "Gagal menjalankan script Python.\nPastikan Python & library terinstall.\nError: " + evaluationTask.getException().getMessage());
            evaluationTask.getException().printStackTrace();
        });

        // --- Jalankan Task ---
        new Thread(evaluationTask).start();
    }

    // --- Fungsi Helper untuk Task Python ---
    private Task<Boolean> createPythonEvaluationTask() {
         return new Task<>() {
            @Override
            protected Boolean call() throws Exception {
                updateMessage("Mempersiapkan perintah Python...");
                updateProgress(-1, 1); // Indeterminate

                // ==========================================================
                // ==== !! PENTING: SESUAIKAN PATH INI !! ====
                // ==========================================================
                // Tentukan path absolut ke script evaluate.py Anda
                String pythonScriptPath = "C:\\Users\\LENOVO\\Documents\\CodeWorkspace\\DockerGenerate\\testing\\evaluate.py"; // <--- GANTI INI!
                // Tentukan perintah python (biasanya "python" atau "python3")
                String pythonCommand = "python";
                // ==========================================================

                File scriptFile = new File(pythonScriptPath);
                if (!scriptFile.exists()) {
                    updateMessage("Error: Script Python tidak ditemukan di " + pythonScriptPath);
                    throw new FileNotFoundException("Script Python tidak ditemukan: " + pythonScriptPath);
                }


                List<String> command = new ArrayList<>();
                command.add(pythonCommand);
                command.add(scriptFile.getAbsolutePath());
                command.add("--model-path");
                command.add(modelFile.getAbsolutePath());
                 command.add("--input-dir");
                command.add(inputDir.getAbsolutePath());
                command.add("--output-dir");
                command.add(outputDir.getAbsolutePath());
                // command.add("--batch-size"); // Tambahkan argumen lain jika perlu
                // command.add("64");

                System.out.println("Executing command: " + String.join(" ", command));

                updateMessage("Menjalankan evaluasi via Python lokal...");
                ProcessBuilder processBuilder = new ProcessBuilder(command);
                processBuilder.redirectErrorStream(true);
                Process process = processBuilder.start();

                // Baca output Python ke konsol Java
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("Python Log: " + line);
                        // updateMessage("Python: " + line); // Bisa update UI jika mau
                    }
                }

                int exitCode = process.waitFor();
                System.out.println("Python process finished with exit code: " + exitCode);
                return exitCode == 0;
            }
        };
    }

    // --- Fungsi Memuat Hasil ---
    private void loadResults() {
        System.out.println("Mencoba memuat hasil dari: " + outputDir.getAbsolutePath());
        File metricsFile = new File(outputDir, "metrics.json");
        File cmFile = new File(outputDir, "confusion_matrix.png");
        File predsFile = new File(outputDir, "predictions.json");
        boolean resultsLoaded = false;

        // 1. Muat Metrik
        if (metricsFile.exists()) {
            try {
                Map<String, Double> metrics = objectMapper.readValue(metricsFile, new TypeReference<>() {});
                lblAkurasi.setText(String.format("%.2f%%", metrics.getOrDefault("accuracy", 0.0)));
                lblPresisi.setText(String.format("%.2f%%", metrics.getOrDefault("precision_macro", 0.0)));
                lblRecall.setText(String.format("%.2f%%", metrics.getOrDefault("recall_macro", 0.0)));
                lblF1.setText(String.format("%.2f%%", metrics.getOrDefault("f1_macro", 0.0)));
                System.out.println("Metrik berhasil dimuat.");
                resultsLoaded = true;
            } catch (IOException e) {
                handleLoadError("metrics.json", e);
            }
        } else {
            System.err.println("File metrics.json tidak ditemukan.");
        }

        // 2. Muat Confusion Matrix
        if (cmFile.exists()) {
            try (FileInputStream cmStream = new FileInputStream(cmFile)) {
                imgConfusionMatrix.setImage(new Image(cmStream));
                System.out.println("Confusion matrix berhasil dimuat.");
                resultsLoaded = true;
            } catch (IOException e) {
                 handleLoadError("confusion_matrix.png", e);
            }
        } else {
            System.err.println("File confusion_matrix.png tidak ditemukan.");
        }

        // 3. Muat Galeri Prediksi
        if (predsFile.exists()) {
            try {
                List<GalleryItem> items = objectMapper.readValue(predsFile, new TypeReference<>() {});
                galleryListView.setItems(FXCollections.observableArrayList(items));
                System.out.println("Daftar prediksi ("+ items.size() +" item) berhasil dimuat.");
                // Cell factory sudah di-setup di initialize
                resultsLoaded = true;
            } catch (IOException e) {
                 handleLoadError("predictions.json", e);
            }
        } else {
             System.err.println("File predictions.json tidak ditemukan.");
        }

        if (!resultsLoaded) {
            showAlert(Alert.AlertType.WARNING, "Warning", "Tidak ada file hasil (JSON/PNG) yang ditemukan atau bisa dimuat dari folder output.");
        }
    }

    // --- Fungsi Helper UI ---

    // Mengosongkan tampilan hasil
    private void clearResultsUI() {
        lblAkurasi.setText("-");
        lblPresisi.setText("-");
        lblRecall.setText("-");
        lblF1.setText("-");
        imgConfusionMatrix.setImage(null);
        galleryListView.setItems(FXCollections.observableArrayList()); // Kosongkan list
    }

    // Menangani error saat load file hasil
    private void handleLoadError(String fileName, Exception e) {
        System.err.println("Gagal membaca " + fileName + ": " + e.getMessage());
        e.printStackTrace();
        showAlert(Alert.AlertType.WARNING, "Warning Memuat Hasil", "Gagal memuat file " + fileName + ".");
    }

    // Setup tampilan galeri
    private void setupGalleryCellFactory() {
         galleryListView.setCellFactory(param -> new ListCell<>() {
            private final HBox hbox = new HBox(10); // HBox(spacing)
            private final ImageView thumbnail = new ImageView();
            private final VBox textVBox = new VBox(); // VBox untuk teks
            private final Label fileNameLabel = new Label();
            private final Label actualLabel = new Label();
            private final Label predictedLabel = new Label();

            // Blok inisialisasi untuk setup awal cell
            {
                fileNameLabel.setStyle("-fx-font-weight: bold;");
                textVBox.getChildren().addAll(fileNameLabel, actualLabel, predictedLabel);

                thumbnail.setFitWidth(60); // Ukuran thumbnail
                thumbnail.setFitHeight(60);
                thumbnail.setPreserveRatio(true);
                thumbnail.setSmooth(true); // Gambar lebih halus

                hbox.getChildren().addAll(thumbnail, textVBox);
                hbox.setAlignment(Pos.CENTER_LEFT);
            }

            @Override
            protected void updateItem(GalleryItem item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setGraphic(null);
                    thumbnail.setImage(null); // Hapus gambar dari cell sebelumnya
                } else {
                    fileNameLabel.setText("File: " + item.fileName);
                    actualLabel.setText("Asli: " + item.actualLabel);
                    predictedLabel.setText("Prediksi: " + item.predictedLabel);

                    // Set warna teks prediksi
                    if (item.actualLabel.equals(item.predictedLabel)) {
                        predictedLabel.setTextFill(Color.GREEN);
                    } else {
                        predictedLabel.setTextFill(Color.RED);
                    }

                    // Muat thumbnail (di-handle oleh fungsi terpisah)
                    loadThumbnailForItemAsync(item);

                    setGraphic(hbox); // Tampilkan HBox di cell
                }
            }

            // Fungsi memuat thumbnail secara asynchronous agar UI tidak freeze
            private void loadThumbnailForItemAsync(GalleryItem item) {
                // Gambar placeholder awal (opsional)
                 thumbnail.setImage(null); // Kosongkan dulu

                 if (inputDir == null || !inputDir.isDirectory()) return;

                 File imageFile = findImageFile(inputDir, item.actualLabel, item.fileName);
                 if (imageFile != null && imageFile.exists()) {
                     // Gunakan Image constructor dengan backgroundLoading=true
                     Image img = new Image(imageFile.toURI().toString(), 60, 60, true, true, true);
                     thumbnail.setImage(img);

                     // Handle jika gagal load (misal file rusak)
                     img.errorProperty().addListener((obs, oldVal, newVal) -> {
                         if (newVal) {
                             System.err.println("Gagal load thumbnail async: " + item.fileName);
                             thumbnail.setImage(null); // Set ke null jika error
                         }
                     });

                 } else {
                      System.err.println("Thumbnail tidak ditemukan: " + item.fileName + " di " + item.actualLabel);
                      thumbnail.setImage(null); // Atau gambar placeholder default
                 }
            }
        });
    }

    // Setup listener klik pada galeri
    private void setupGalleryClickListener() {
        galleryListView.getSelectionModel().selectedItemProperty().addListener((obs, oldSel, newSel) -> {
            if (newSel != null) {
                Alert detailDialog = new Alert(Alert.AlertType.INFORMATION);
                detailDialog.setTitle("Detail Prediksi Gambar");
                detailDialog.setHeaderText("File: " + newSel.fileName);

                // Muat Gambar Besar dari Folder INPUT
                ImageView dialogImage = new ImageView();
                dialogImage.setFitHeight(250); // Ukuran gambar detail
                dialogImage.setPreserveRatio(true);
                dialogImage.setSmooth(true);

                 if (inputDir != null && inputDir.isDirectory()) {
                    File imageFile = findImageFile(inputDir, newSel.actualLabel, newSel.fileName);
                    if (imageFile != null && imageFile.exists()) {
                        // Muat gambar secara async untuk dialog
                        Image img = new Image(imageFile.toURI().toString(), 0, 250, true, true, true);
                        dialogImage.setImage(img);
                        img.errorProperty().addListener((ob, ov, nv) -> {
                             if (nv) System.err.println("Gagal load gambar detail: " + newSel.fileName);
                        });
                        detailDialog.setGraphic(dialogImage);
                    } else {
                        System.err.println("Gambar detail tidak ditemukan: " + newSel.fileName);
                        detailDialog.setGraphic(null); // Tidak ada gambar jika tidak ketemu
                    }
                } else {
                     detailDialog.setGraphic(null);
                 }

                // Buat Teks Konten Detail
                StringBuilder content = new StringBuilder();
                content.append(String.format("Prediksi Utama: %s (Confidence: %.2f%%)\n",
                        newSel.predictedLabel, newSel.confidence));
                content.append("Label Asli: " + newSel.actualLabel + "\n\n");
                content.append("--- Skor Prediksi Semua Kelas ---\n");

                // Urutkan skor dari tertinggi
                newSel.allScores.entrySet().stream()
                        .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                        .forEach(entry -> content.append(String.format("%s: %.2f%%\n", entry.getKey(), entry.getValue())));

                TextArea textArea = new TextArea(content.toString());
                textArea.setEditable(false);
                textArea.setWrapText(true);
                textArea.setPrefHeight(200);

                detailDialog.getDialogPane().setContent(textArea);
                detailDialog.getDialogPane().setPrefWidth(550); // Lebarkan dialog
                detailDialog.setResizable(true);

                detailDialog.showAndWait();
                galleryListView.getSelectionModel().clearSelection(); // Hapus seleksi
            }
        });
    }

    // Fungsi helper mencari file gambar
    private File findImageFile(File baseDir, String className, String fileName) {
        File classDir = new File(baseDir, className);
        if (classDir.isDirectory()) {
            File imgFile = new File(classDir, fileName);
            if (imgFile.exists()) return imgFile;
        }
        // Fallback: coba cari di baseDir jika tidak ketemu di subfolder
        File imgFileRoot = new File(baseDir, fileName);
        if (imgFileRoot.exists()) return imgFileRoot;

        return null; // Tidak ketemu
    }

    // Fungsi helper menampilkan Alert
    private void showAlert(Alert.AlertType type, String title, String message) {
        // Jalankan di JavaFX Application Thread jika dipanggil dari thread background
        if (!javafx.application.Platform.isFxApplicationThread()) {
            javafx.application.Platform.runLater(() -> showAlertInternal(type, title, message));
        } else {
            showAlertInternal(type, title, message);
        }
    }
    private void showAlertInternal(Alert.AlertType type, String title, String message) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}