package eyepred.eyediseasesclassifier; // Ganti dengan package Anda

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.geometry.Pos;
import javafx.scene.Scene; // Import BARU
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent; // Import BARU
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane; // Import BARU
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.Modality; // Import BARU
import javafx.stage.Stage; // Import BARU

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MainController {

    // === Path ke Skrip Python (GANTI INI) ===
    // Path ini diambil dari file MainController.java yang Anda unggah
    private static final String PYTHON_EXECUTABLE = "C:\\Users\\LENOVO\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"; //
    private static final String PYTHON_SCRIPT_PATH = "C:\\Users\\LENOVO\\Documents\\CodeWorkspace\\DockerGenerate\\testing\\train_and_evaluate.py"; //

    // === Injeksi FXML (Panel Kontrol) ===
    @FXML private Button btnInputDataset;
    @FXML private Label lblInputPath;
    @FXML private CheckBox chkBalanceData;
    @FXML private TextField txtSplitTrain;
    @FXML private TextField txtSplitVal;
    @FXML private TextField txtSplitTest;
    @FXML private CheckBox chkFreezeBase;
    @FXML private TextField txtDenseNeurons;
    @FXML private TextField txtDropoutRate;
    @FXML private TextField txtEpochs;
    @FXML private Button btnOutputDir;
    @FXML private Label lblOutputPath;
    @FXML private Button btnStartTraining;

    // === Injeksi FXML (Panel Output) ===
    // @FXML private TabPane tabPaneOutput; // Dihapus
    @FXML private LineChart<Number, Number> chartTraining;
    @FXML private Label lblAkurasi;
    @FXML private Label lblPresisi;
    @FXML private Label lblRecall;
    @FXML private Label lblF1;
    @FXML private ImageView imgConfusionMatrix;
    @FXML private TextArea txtClassificationReport;
    @FXML private TextArea txtLog;

    // === Injeksi FXML (Galeri) ===
    @FXML private ListView<GalleryItem> galleryListView;

    // === Injeksi FXML (Loading) ===
    @FXML private VBox loadingPane;
    @FXML private Label lblLoadingStatus;
    @FXML private ProgressBar progressBar;
    @FXML private Button btnBatal; // <-- BARU

    // === Variabel Internal ===
    private File inputDir;
    private File outputDir;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private XYChart.Series<Number, Number> accuracySeries;
    private XYChart.Series<Number, Number> lossSeries;

    // === BARU: Variabel untuk Task & Process Control ===
    private Task<Boolean> runningTask;
    private Process runningPythonProcess;

    // === Model data untuk galeri ===
    public static class GalleryItem {
        public String fileName;
        public String actualLabel;
        public String predictedLabel;
        public double confidence;
        public Map<String, Double> allScores;
        public GalleryItem() {}
    }

    @FXML
    public void initialize() {
        // Setup Grafik
        accuracySeries = new XYChart.Series<>();
        accuracySeries.setName("Validation Accuracy");
        lossSeries = new XYChart.Series<>();
        lossSeries.setName("Validation Loss");
        chartTraining.getData().addAll(accuracySeries, lossSeries);

        loadingPane.setVisible(false);

        // Setup fungsionalitas galeri
        setupGalleryCellFactory();
        setupGalleryClickListener();
    }

    @FXML
    private void handleInputDataset() {
        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Pilih Folder Dataset Input");
        File dir = dirChooser.showDialog(btnInputDataset.getScene().getWindow());
        if (dir != null) {
            inputDir = dir;
            lblInputPath.setText(inputDir.getAbsolutePath());
        }
    }

    @FXML
    private void handleOutputFolder() {
        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Pilih Folder Output Hasil");
        File dir = dirChooser.showDialog(btnOutputDir.getScene().getWindow());
        if (dir != null) {
            outputDir = dir;
            lblOutputPath.setText(outputDir.getAbsolutePath());
        }
    }

    @FXML
    private void handleStartTraining() {
        // 1. Validasi Input
        if (inputDir == null || outputDir == null) {
            showAlert(Alert.AlertType.ERROR, "Error", "Folder Input dan Output tidak boleh kosong.");
            return;
        }
        int trainSplit, valSplit, testSplit, epochs;
        float dropout;
        try {
            trainSplit = Integer.parseInt(txtSplitTrain.getText());
            valSplit = Integer.parseInt(txtSplitVal.getText());
            testSplit = Integer.parseInt(txtSplitTest.getText());
            if (trainSplit + valSplit + testSplit != 100) {
                showAlert(Alert.AlertType.ERROR, "Error", "Total split ratio (Train + Val + Test) harus 100.");
                return;
            }
            epochs = Integer.parseInt(txtEpochs.getText());
            dropout = Float.parseFloat(txtDropoutRate.getText());
        } catch (NumberFormatException e) {
            showAlert(Alert.AlertType.ERROR, "Error", "Input Epoch, Split, dan Dropout harus berupa angka valid.");
            return;
        }

        // 2. Bersihkan UI
        clearResultsUI();

        // 3. Buat Perintah Python
        List<String> command = buildPythonCommand(trainSplit, valSplit, testSplit, epochs, dropout);

        // 4. Buat Task Background
        Task<Boolean> trainingTask = createTrainingTask(command);

        // 5. Ikat UI ke Task
        lblLoadingStatus.textProperty().bind(trainingTask.messageProperty());
        progressBar.progressProperty().bind(trainingTask.progressProperty());

        trainingTask.setOnRunning(e -> {
            loadingPane.setVisible(true);
            btnStartTraining.setDisable(true);
        });

        trainingTask.setOnSucceeded(e -> {
            loadingPane.setVisible(false);
            btnStartTraining.setDisable(false);
            if (trainingTask.getValue()) {
                showAlert(Alert.AlertType.INFORMATION, "Sukses", "Training dan evaluasi selesai. Memuat hasil...");
                loadResults();
            } else {
                if (!trainingTask.isCancelled()) {
                     showAlert(Alert.AlertType.ERROR, "Gagal", "Proses Python gagal. Periksa log di bagian atas.");
                }
            }
            runningTask = null;
            runningPythonProcess = null;
        });

        trainingTask.setOnFailed(e -> {
            loadingPane.setVisible(false);
            btnStartTraining.setDisable(false);
            if (!trainingTask.isCancelled()) {
                showAlert(Alert.AlertType.ERROR, "Error Kritis", "Gagal menjalankan task: " + trainingTask.getException().getMessage());
                trainingTask.getException().printStackTrace();
            }
            runningTask = null;
            runningPythonProcess = null;
        });

        // === BARU: Handler untuk Batal ===
        trainingTask.setOnCancelled(e -> {
            loadingPane.setVisible(false);
            btnStartTraining.setDisable(false);
            txtLog.appendText("\n--- PROSES DIBATALKAN OLEH PENGGUNA ---\n");
            showAlert(Alert.AlertType.WARNING, "Dibatalkan", "Proses training telah dibatalkan.");
            
            runningTask = null;
            runningPythonProcess = null;
        });


        // 6. Simpan referensi dan Jalankan Task
        this.runningTask = trainingTask;
        new Thread(this.runningTask).start();
    }
    
    /**
     * === BARU: Handler untuk tombol Batal ===
     */
    @FXML
    private void handleBatal() {
        txtLog.appendText("\n--- MEMBATALKAN PROSES... ---\n");
        
        // Hancurkan proses Python
        if (this.runningPythonProcess != null) {
            // destroyForcibly() membunuh proses dan sub-prosesnya (penting untuk Python)
            this.runningPythonProcess.destroyForcibly(); 
        }
        
        // Batalkan Task JavaFX
        if (this.runningTask != null) {
            this.runningTask.cancel(true); // Mengirim interrupt ke thread task
        }
    }
    
    /**
     * === BARU: Handler untuk Zoom Confusion Matrix ===
     */
    @FXML
    private void handleZoomConfusionMatrix(MouseEvent event) {
        if (imgConfusionMatrix.getImage() == null) {
            return;
        }

        // Buat ImageView baru untuk dialog
        ImageView zoomView = new ImageView(imgConfusionMatrix.getImage());
        zoomView.setPreserveRatio(true);

        // Buat ScrollPane agar bisa di-pan jika gambar lebih besar dari window
        ScrollPane scrollPane = new ScrollPane(zoomView);
        scrollPane.setPannable(true);
        scrollPane.setStyle("-fx-background: #333;"); // Latar belakang gelap

        // StackPane untuk menengahkan gambar di dalam ScrollPane
        StackPane zoomLayout = new StackPane(scrollPane);
        zoomLayout.setStyle("-fx-background-color: #333;");
        
        // Tentukan ukuran awal window pop-up
        double width = Math.min(800, imgConfusionMatrix.getImage().getWidth() + 40);
        double height = Math.min(700, imgConfusionMatrix.getImage().getHeight() + 40);

        Scene zoomScene = new Scene(zoomLayout, width, height);

        Stage zoomStage = new Stage();
        zoomStage.setTitle("Confusion Matrix - Zoom");
        zoomStage.initModality(Modality.APPLICATION_MODAL); // Blok window utama
        zoomStage.initOwner(btnStartTraining.getScene().getWindow()); // Set owner
        zoomStage.setScene(zoomScene);
        zoomStage.showAndWait();
    }


    private List<String> buildPythonCommand(int train, int val, int test, int epochs, float dropout) {
        List<String> command = new ArrayList<>();
        command.add(PYTHON_EXECUTABLE);
        command.add(PYTHON_SCRIPT_PATH);
        command.add("--input-dir");
        command.add(inputDir.getAbsolutePath());
        command.add("--output-dir");
        command.add(outputDir.getAbsolutePath());
        command.add("--split-ratio");
        command.add(String.format("%d,%d,%d", train, val, test));
        command.add("--epochs");
        command.add(String.valueOf(epochs));
        command.add("--dropout-rate");
        command.add(String.valueOf(dropout));
        
        String denseNeurons = txtDenseNeurons.getText().trim();
        if (!denseNeurons.isEmpty()) {
             command.add("--dense-neurons");
             command.add(denseNeurons);
        }
        if (chkBalanceData.isSelected()) {
            command.add("--balance-data");
        }
        if (chkFreezeBase.isSelected()) {
            command.add("--freeze-base");
        }
        
        System.out.println("Executing command: " + String.join(" ", command));
        return command;
    }

    private Task<Boolean> createTrainingTask(List<String> command) {
        return new Task<>() {
            @Override
            protected Boolean call() throws Exception {
                int totalEpochs = 1;
                for(int i=0; i < command.size(); i++) {
                    if(command.get(i).equals("--epochs") && i+1 < command.size()) {
                        totalEpochs = Integer.parseInt(command.get(i+1));
                        break;
                    }
                }
                
                ProcessBuilder processBuilder = new ProcessBuilder(command);
                processBuilder.redirectErrorStream(true);
                
                // Simpan referensi proses
                runningPythonProcess = processBuilder.start(); 

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(runningPythonProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        // Jika task dibatalkan, berhenti membaca
                        if (isCancelled()) {
                            break;
                        }
                        
                        final String logLine = line;
                        Platform.runLater(() -> txtLog.appendText(logLine + "\n"));

                        if (logLine.trim().startsWith("{\"epoch\":")) {
                            try {
                                Map<String, Object> epochData = objectMapper.readValue(logLine, new TypeReference<>() {});
                                int epoch = (Integer) epochData.get("epoch") + 1;
                                double acc = ((Number) epochData.get("val_accuracy")).doubleValue();
                                double loss = ((Number) epochData.get("val_loss")).doubleValue();

                                final int currentEpoch = epoch;
                                final int finalTotalEpochs = totalEpochs;
                                Platform.runLater(() -> {
                                    updateMessage(String.format("Training Epoch %d/%d...", currentEpoch, finalTotalEpochs));
                                    updateProgress(currentEpoch, finalTotalEpochs);
                                    accuracySeries.getData().add(new XYChart.Data<>(currentEpoch, acc));
                                    lossSeries.getData().add(new XYChart.Data<>(currentEpoch, loss));
                                });

                            } catch (Exception e) {
                                if (!isCancelled()) {
                                    System.err.println("Gagal parse JSON real-time: " + e.getMessage());
                                }
                            }
                        }
                    }
                }

                int exitCode = runningPythonProcess.waitFor();
                runningPythonProcess = null; 
                
                if (isCancelled()) {
                    return false; 
                }
                
                return exitCode == 0;
            }
        };
    }
    
    private void clearResultsUI() {
        accuracySeries.getData().clear();
        lossSeries.getData().clear();
        lblAkurasi.setText("-");
        lblPresisi.setText("-");
        lblRecall.setText("-");
        lblF1.setText("-");
        imgConfusionMatrix.setImage(null);
        txtClassificationReport.clear();
        txtLog.clear(); // Hapus log sebelumnya

        if (galleryListView != null) {
            galleryListView.setItems(FXCollections.observableArrayList());
        }
    }

    private void loadResults() {
        if (outputDir == null) return;
        System.out.println("Mencoba memuat hasil dari: " + outputDir.getAbsolutePath());

        File metricsFile = new File(outputDir, "final_metrics.json");
        File cmFile = new File(outputDir, "confusion_matrix_test.png");
        File reportFile = new File(outputDir, "classification_report_test.txt");
        File modelFile = new File(outputDir, "trained_model.h5");
        File predsFile = new File(outputDir, "predictions.json"); 

        // 1. Muat Metrik
        if (metricsFile.exists()) {
            try {
                Map<String, Object> metrics = objectMapper.readValue(metricsFile, new TypeReference<>() {});
                lblAkurasi.setText(String.format("%.2f%%", ((Number) metrics.getOrDefault("accuracy", 0.0)).doubleValue() * 100));
                lblPresisi.setText(String.format("%.2f%%", ((Number) metrics.getOrDefault("precision_macro", 0.0)).doubleValue() * 100));
                lblRecall.setText(String.format("%.2f%%", ((Number) metrics.getOrDefault("recall_macro", 0.0)).doubleValue() * 100));
                lblF1.setText(String.format("%.2f%%", ((Number) metrics.getOrDefault("f1_macro", 0.0)).doubleValue() * 100));
            } catch (IOException e) {
                System.err.println("Gagal baca metrics.json: " + e.getMessage());
            }
        } else {
             System.err.println("File final_metrics.json tidak ditemukan.");
        }

        // 2. Muat Confusion Matrix
        if (cmFile.exists()) {
            try (FileInputStream cmStream = new FileInputStream(cmFile)) {
                imgConfusionMatrix.setImage(new Image(cmStream));
            } catch (IOException e) {
                System.err.println("Gagal baca confusion_matrix_test.png: " + e.getMessage());
            }
        } else {
             System.err.println("File confusion_matrix_test.png tidak ditemukan.");
        }

        // 3. Muat Classification Report
        if (reportFile.exists()) {
            try {
                String reportText = new String(Files.readAllBytes(Paths.get(reportFile.getAbsolutePath())), StandardCharsets.UTF_8);
                txtClassificationReport.setText(reportText);
            } catch (IOException e) {
                 System.err.println("Gagal baca classification_report_test.txt: " + e.getMessage());
            }
        } else {
            System.err.println("File classification_report_test.txt tidak ditemukan.");
            txtClassificationReport.setText("File laporan tidak ditemukan.");
        }

        // 4. Muat Galeri Prediksi
        if (predsFile.exists()) {
            try {
                List<GalleryItem> items = objectMapper.readValue(predsFile, new TypeReference<>() {});
                galleryListView.setItems(FXCollections.observableArrayList(items));
                System.out.println("Daftar prediksi galeri ("+ items.size() +" item) berhasil dimuat.");
            } catch (IOException e) {
                 System.err.println("Gagal baca predictions.json: " + e.getMessage());
            }
        } else {
             System.err.println("File predictions.json tidak ditemukan.");
        }

        // 5. Konfirmasi Model Disimpan
        if (modelFile.exists()) {
            txtLog.appendText(String.format("\n--- MODEL BERHASIL DISIMPAN ---\n%s\n", modelFile.getAbsolutePath()));
        }
    }

    // ==========================================================
    // === FUNGSI-FUNGSI GALERI (Tidak Berubah) ===
    // ==========================================================
    
    private void setupGalleryCellFactory() {
         galleryListView.setCellFactory(param -> new ListCell<>() {
            private final HBox hbox = new HBox(10);
            private final ImageView thumbnail = new ImageView();
            private final VBox textVBox = new VBox();
            private final Label fileNameLabel = new Label();
            private final Label actualLabel = new Label();
            private final Label predictedLabel = new Label();

            {
                fileNameLabel.setStyle("-fx-font-weight: bold;");
                textVBox.getChildren().addAll(fileNameLabel, actualLabel, predictedLabel);
                thumbnail.setFitWidth(60);
                thumbnail.setFitHeight(60);
                thumbnail.setPreserveRatio(true);
                thumbnail.setSmooth(true);
                hbox.getChildren().addAll(thumbnail, textVBox);
                hbox.setAlignment(Pos.CENTER_LEFT);
            }

            @Override
            protected void updateItem(GalleryItem item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setText(null);
                    setGraphic(null);
                    thumbnail.setImage(null);
                } else {
                    fileNameLabel.setText("File: " + item.fileName);
                    actualLabel.setText("Asli: " + item.actualLabel);
                    predictedLabel.setText("Prediksi: " + item.predictedLabel);

                    if (item.actualLabel.equals(item.predictedLabel)) {
                        predictedLabel.setTextFill(Color.GREEN);
                    } else {
                        predictedLabel.setTextFill(Color.RED);
                    }
                    loadThumbnailForItemAsync(item);
                    setGraphic(hbox);
                }
            }
            
            private void loadThumbnailForItemAsync(GalleryItem item) {
                 thumbnail.setImage(null);
                 if (inputDir == null || !inputDir.isDirectory()) return;
                 File imageFile = findImageFile(inputDir, item.actualLabel, item.fileName);
                 
                 if (imageFile != null && imageFile.exists()) {
                     Image img = new Image(imageFile.toURI().toString(), 60, 60, true, true, true);
                     thumbnail.setImage(img);
                     img.errorProperty().addListener((obs, oldVal, newVal) -> {
                         if (newVal) {
                             System.err.println("Gagal load thumbnail async: " + item.fileName);
                             thumbnail.setImage(null);
                         }
                     });
                 } else {
                     System.err.println("Thumbnail tidak ditemukan: " + item.fileName + " di " + item.actualLabel);
                     thumbnail.setImage(null);
                 }
            }
        });
    }

    private void setupGalleryClickListener() {
        galleryListView.getSelectionModel().selectedItemProperty().addListener((obs, oldSel, newSel) -> {
            if (newSel != null) {
                Alert detailDialog = new Alert(Alert.AlertType.INFORMATION);
                detailDialog.setTitle("Detail Prediksi Gambar");
                detailDialog.setHeaderText("File: " + newSel.fileName);

                ImageView dialogImage = new ImageView();
                dialogImage.setFitHeight(250);
                dialogImage.setPreserveRatio(true);
                dialogImage.setSmooth(true);

                 if (inputDir != null && inputDir.isDirectory()) {
                     File imageFile = findImageFile(inputDir, newSel.actualLabel, newSel.fileName);
                     if (imageFile != null && imageFile.exists()) {
                         Image img = new Image(imageFile.toURI().toString(), 0, 250, true, true, true);
                         dialogImage.setImage(img);
                         img.errorProperty().addListener((ob, ov, nv) -> {
                             if (nv) System.err.println("Gagal load gambar detail: " + newSel.fileName);
                         });
                         detailDialog.setGraphic(dialogImage);
                     } else {
                         System.err.println("Gambar detail tidak ditemukan: " + newSel.fileName);
                         detailDialog.setGraphic(null);
                     }
                 } else {
                     detailDialog.setGraphic(null);
                 }

                StringBuilder content = new StringBuilder();
                content.append(String.format("Prediksi Utama: %s (Confidence: %.2f%%)\n",
                        newSel.predictedLabel, newSel.confidence));
                content.append("Label Asli: " + newSel.actualLabel + "\n\n");
                content.append("--- Skor Prediksi Semua Kelas ---\n");

                newSel.allScores.entrySet().stream()
                        .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                        .forEach(entry -> content.append(String.format("%s: %.2f%%\n", entry.getKey(), entry.getValue())));

                TextArea textArea = new TextArea(content.toString());
                textArea.setEditable(false);
                textArea.setWrapText(true);
                textArea.setPrefHeight(200);

                detailDialog.getDialogPane().setContent(textArea);
                detailDialog.getDialogPane().setPrefWidth(550);
                detailDialog.setResizable(true);

                detailDialog.showAndWait();
                galleryListView.getSelectionModel().clearSelection();
            }
        });
    }

    private File findImageFile(File baseDir, String className, String fileName) {
        if (baseDir == null || className == null || fileName == null) return null;
        
        File classDir = new File(baseDir, className);
        if (classDir.isDirectory()) {
            File imgFile = new File(classDir, fileName);
            if (imgFile.exists()) return imgFile;
        }
        
        File imgFileRoot = new File(baseDir, fileName);
        if (imgFileRoot.exists()) return imgFileRoot;

        return null;
    }

    // ==========================================================
    // === FUNGSI HELPER ALERT (Tidak Berubah) ===
    // ==========================================================

    private void showAlert(Alert.AlertType type, String title, String message) {
        if (!Platform.isFxApplicationThread()) {
            Platform.runLater(() -> showAlertInternal(type, title, message));
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