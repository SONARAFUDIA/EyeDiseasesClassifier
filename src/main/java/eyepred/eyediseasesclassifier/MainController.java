package eyepred.eyediseasesclassifier;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MainController {

    // === Path ke Skrip Python ===
    // Pastikan path ini benar di komputer kamu!
    private static final String PYTHON_EXECUTABLE = "C:\\Users\\LENOVO\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"; 
    
    // Arahkan ke script train_and_evaluate.py yang SUDAH kamu update di repo
    private static final String PYTHON_SCRIPT_PATH = "C:\\Users\\LENOVO\\Documents\\NetBeansProjects\\EyeDiseasesClassifier\\train_and_evaluate.py";

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
    @FXML private Button btnBatal;

    // === Injeksi FXML (Load Model) ===
    @FXML private Button btnLoadModel;
    @FXML private Button btnClearModel;
    @FXML private Label lblLoadModelPath;
    @FXML private VBox vboxVggConfig;

    // === Injeksi FXML (Panel Output) ===
    @FXML private LineChart<Number, Number> chartTraining;
    @FXML private Label lblAkurasi;
    @FXML private Label lblPresisi;
    @FXML private Label lblRecall;
    @FXML private Label lblF1;
    @FXML private ImageView imgConfusionMatrix;
    @FXML private TextArea txtClassificationReport;
    @FXML private TextArea txtLog;

    // === Injeksi FXML (Galeri & Loading) ===
    @FXML private ListView<GalleryItem> galleryListView;
    @FXML private VBox loadingPane;
    @FXML private Label lblLoadingStatus;
    @FXML private ProgressBar progressBar;

    // === Variabel Internal ===
    private File inputDir;
    private File outputDir;
    private File loadModelFile = null; // Menyimpan path model untuk dilanjutkan
    private final ObjectMapper objectMapper = new ObjectMapper();
    private XYChart.Series<Number, Number> accuracySeries;
    private XYChart.Series<Number, Number> lossSeries;

    // === Variabel Task ===
    private Task<Boolean> runningTask;
    private Process runningPythonProcess;

    // === Model Galeri ===
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
        accuracySeries = new XYChart.Series<>();
        accuracySeries.setName("Validation Accuracy");
        lossSeries = new XYChart.Series<>();
        lossSeries.setName("Validation Loss");
        chartTraining.getData().addAll(accuracySeries, lossSeries);

        loadingPane.setVisible(false);
        btnBatal.setDisable(true);
        
        handleClearLoadModel(); // Set state awal (Mode Train Baru)
        
        setupGalleryCellFactory();
        setupGalleryClickListener();
    }

    @FXML
    private void handleInputDataset() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Pilih Folder Dataset");
        File f = dc.showDialog(btnInputDataset.getScene().getWindow());
        if (f != null) {
            inputDir = f;
            lblInputPath.setText(f.getAbsolutePath());
            loadStaticConfusionMatrix(inputDir); // Load gambar statis jika ada
        }
    }

    private void loadStaticConfusionMatrix(File directory) {
        File cmFile = new File(directory, "confusion_matrix.png");
        if (cmFile.exists()) {
            try {
                imgConfusionMatrix.setImage(new Image(new FileInputStream(cmFile)));
            } catch (Exception e) { e.printStackTrace(); }
        }
    }

    @FXML
    private void handleOutputFolder() {
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Pilih Folder Output");
        File f = dc.showDialog(btnOutputDir.getScene().getWindow());
        if (f != null) { outputDir = f; lblOutputPath.setText(f.getAbsolutePath()); }
    }

    @FXML
    private void handleLoadModel() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Pilih Model .h5");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Model Keras", "*.h5"));
        File f = fc.showOpenDialog(btnLoadModel.getScene().getWindow());
        if (f != null) {
            loadModelFile = f;
            lblLoadModelPath.setText("Lanjut dari: " + f.getName());
            vboxVggConfig.setDisable(true); // Nonaktifkan config VGG karena pakai model lama
        }
    }

    @FXML
    private void handleClearLoadModel() {
        loadModelFile = null;
        lblLoadModelPath.setText("Train dari awal (VGG16)");
        vboxVggConfig.setDisable(false); // Aktifkan config VGG untuk training baru
    }

    @FXML
    private void handleStartTraining() {
        if (inputDir == null || outputDir == null) {
            showAlert(Alert.AlertType.ERROR, "Error", "Pilih folder input & output!");
            return;
        }
        
        try {
            Integer.parseInt(txtEpochs.getText());
        } catch (Exception e) {
            showAlert(Alert.AlertType.ERROR, "Error", "Epoch harus angka!");
            return;
        }

        // 1. Bersihkan UI (TAPI JANGAN HAPUS PILIHAN MODEL!)
        clearResultsUI(); 
        
        // 2. Susun Perintah Python
        List<String> cmd = new ArrayList<>();
        cmd.add(PYTHON_EXECUTABLE);
        cmd.add(PYTHON_SCRIPT_PATH);
        cmd.add("--input-dir"); cmd.add(inputDir.getAbsolutePath());
        cmd.add("--output-dir"); cmd.add(outputDir.getAbsolutePath());
        cmd.add("--epochs"); cmd.add(txtEpochs.getText());
        cmd.add("--split-ratio"); cmd.add(txtSplitTrain.getText()+","+txtSplitVal.getText()+","+txtSplitTest.getText());
        
        // Cek apakah user memilih file model untuk dilanjutkan
        if (loadModelFile != null) {
            cmd.add("--load-model-path"); 
            cmd.add(loadModelFile.getAbsolutePath());
            System.out.println("INFO: Mode Lanjutkan Training.");
        } else {
            // Jika training baru, kirim parameter VGG16
            if (!txtDenseNeurons.getText().isEmpty()) { 
                cmd.add("--dense-neurons"); 
                cmd.add(txtDenseNeurons.getText()); 
            }
            cmd.add("--dropout-rate"); cmd.add(txtDropoutRate.getText());
            if (chkFreezeBase.isSelected()) cmd.add("--freeze-base");
            System.out.println("INFO: Mode Training Baru.");
        }
        
        if (chkBalanceData.isSelected()) cmd.add("--balance-data");

        System.out.println("CMD: " + String.join(" ", cmd));

        // 3. Jalankan Task Background
        Task<Boolean> task = new Task<>() {
            @Override protected Boolean call() throws Exception {
                ProcessBuilder pb = new ProcessBuilder(cmd);
                pb.redirectErrorStream(true);
                runningPythonProcess = pb.start();
                try (BufferedReader r = new BufferedReader(new InputStreamReader(runningPythonProcess.getInputStream()))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        if (isCancelled()) break;
                        final String l = line;
                        Platform.runLater(() -> {
                            txtLog.appendText(l + "\n");
                            // Update Grafik Real-time
                            if (l.trim().startsWith("{\"epoch\":")) {
                                try {
                                    Map<String,Object> d = objectMapper.readValue(l, new TypeReference<>(){});
                                    int ep = (Integer)d.get("epoch")+1;
                                    accuracySeries.getData().add(new XYChart.Data<>(ep, ((Number)d.get("val_accuracy")).doubleValue()));
                                    lossSeries.getData().add(new XYChart.Data<>(ep, ((Number)d.get("val_loss")).doubleValue()));
                                } catch(Exception e){}
                            }
                        });
                    }
                }
                return runningPythonProcess.waitFor() == 0 && !isCancelled();
            }
        };

        // Handler Status Task
        task.setOnRunning(e -> { 
            loadingPane.setVisible(true); 
            btnStartTraining.setDisable(true); 
            btnBatal.setDisable(false); 
        });
        
        task.setOnSucceeded(e -> {
            loadingPane.setVisible(false); 
            btnStartTraining.setDisable(false); 
            btnBatal.setDisable(true);
            
            if (task.getValue()) { 
                showAlert(Alert.AlertType.INFORMATION, "Sukses", "Training Selesai!"); 
                loadResults(); 
            } else if (!task.isCancelled()) {
                showAlert(Alert.AlertType.ERROR, "Gagal", "Proses Python gagal. Cek log.");
            }
            cleanupTask();
        });
        
        task.setOnFailed(e -> {
            loadingPane.setVisible(false); 
            btnStartTraining.setDisable(false); 
            btnBatal.setDisable(true);
            if (!task.isCancelled()) {
                showAlert(Alert.AlertType.ERROR, "Error", task.getException().getMessage());
                task.getException().printStackTrace();
            }
            cleanupTask();
        });
        
        task.setOnCancelled(e -> {
            loadingPane.setVisible(false); 
            btnStartTraining.setDisable(false); 
            btnBatal.setDisable(true);
            txtLog.appendText("\n--- DIBATALKAN OLEH PENGGUNA ---\n");
            cleanupTask();
        });

        runningTask = task;
        new Thread(task).start();
    }

    private void cleanupTask() {
        runningTask = null;
        runningPythonProcess = null;
    }

    @FXML
    private void handleBatal() {
        if (runningPythonProcess != null) runningPythonProcess.destroyForcibly();
        if (runningTask != null) runningTask.cancel(true);
    }

    @FXML
    private void handleZoomConfusionMatrix(MouseEvent e) {
        if (imgConfusionMatrix.getImage() == null) return;
        ImageView iv = new ImageView(imgConfusionMatrix.getImage()); 
        iv.setPreserveRatio(true);
        
        ScrollPane sp = new ScrollPane(new StackPane(iv));
        sp.setPannable(true);
        sp.setStyle("-fx-background: #333;");
        
        Stage s = new Stage(); 
        s.initModality(Modality.APPLICATION_MODAL);
        s.setTitle("Confusion Matrix Zoom");
        s.setScene(new Scene(sp, 800, 600));
        s.showAndWait();
    }

    private void loadResults() {
        try {
            File m = new File(outputDir, "final_metrics.json");
            if (m.exists()) {
                Map<String,Object> d = objectMapper.readValue(m, new TypeReference<>(){});
                lblAkurasi.setText(String.format("%.2f%%", ((Number)d.get("accuracy")).doubleValue()*100));
                lblPresisi.setText(String.format("%.2f%%", ((Number)d.get("precision_macro")).doubleValue()*100));
                lblRecall.setText(String.format("%.2f%%", ((Number)d.get("recall_macro")).doubleValue()*100));
                lblF1.setText(String.format("%.2f%%", ((Number)d.get("f1_macro")).doubleValue()*100));
            }
            File cm = new File(outputDir, "confusion_matrix_test.png");
            if (cm.exists()) imgConfusionMatrix.setImage(new Image(new FileInputStream(cm)));
            
            File r = new File(outputDir, "classification_report_test.txt");
            if (r.exists()) txtClassificationReport.setText(Files.readString(r.toPath()));

            File p = new File(outputDir, "predictions.json");
            if (p.exists()) {
                List<GalleryItem> l = objectMapper.readValue(p, new TypeReference<>(){});
                galleryListView.setItems(FXCollections.observableArrayList(l));
            }
        } catch(Exception e) { e.printStackTrace(); }
    }
    
    private void clearResultsUI() {
        accuracySeries.getData().clear(); 
        lossSeries.getData().clear();
        lblAkurasi.setText("-"); 
        lblPresisi.setText("-"); 
        lblRecall.setText("-"); 
        lblF1.setText("-");
        txtClassificationReport.clear(); 
        txtLog.clear();
        if (galleryListView != null) galleryListView.getItems().clear();
    }

    private void setupGalleryCellFactory() {
        galleryListView.setCellFactory(p -> new ListCell<>() {
            HBox hb = new HBox(10); ImageView iv = new ImageView(); VBox vb = new VBox();
            Label lName = new Label(), lAct = new Label(), lPred = new Label();
            {
                lName.setStyle("-fx-font-weight:bold"); vb.getChildren().addAll(lName, lAct, lPred);
                iv.setFitHeight(60); iv.setFitWidth(60); iv.setPreserveRatio(true);
                hb.getChildren().addAll(iv, vb); hb.setAlignment(Pos.CENTER_LEFT);
            }
            @Override protected void updateItem(GalleryItem item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) { setText(null); setGraphic(null); }
                else {
                    lName.setText(item.fileName); lAct.setText("Asli: "+item.actualLabel);
                    lPred.setText("Prediksi: "+item.predictedLabel);
                    lPred.setTextFill(item.actualLabel.equals(item.predictedLabel) ? Color.GREEN : Color.RED);
                    if (inputDir != null) {
                        File f = findImageFile(inputDir, item.actualLabel, item.fileName);
                        if (f != null && f.exists()) {
                            try { iv.setImage(new Image(new FileInputStream(f))); } catch(Exception e){}
                        }
                    }
                    setGraphic(hb);
                }
            }
        });
    }
    
    private File findImageFile(File baseDir, String label, String fileName) {
        File f = new File(new File(baseDir, label), fileName);
        if (f.exists()) return f;
        f = new File(baseDir, fileName);
        if (f.exists()) return f;
        return null;
    }

    private void setupGalleryClickListener() {
        galleryListView.getSelectionModel().selectedItemProperty().addListener((o,old,item)->{
            if(item==null) return;
            Alert a = new Alert(Alert.AlertType.INFORMATION); a.setHeaderText(item.fileName);
            StringBuilder sb = new StringBuilder("Scores:\n");
            item.allScores.forEach((k,v)->sb.append(k).append(": ").append(String.format("%.2f",v)).append("%\n"));
            a.setContentText(sb.toString()); a.showAndWait();
        });
    }

    private void showAlert(Alert.AlertType type, String title, String msg) {
        Platform.runLater(() -> {
            Alert a = new Alert(type); a.setTitle(title); a.setContentText(msg); a.showAndWait();
        });
    }
}