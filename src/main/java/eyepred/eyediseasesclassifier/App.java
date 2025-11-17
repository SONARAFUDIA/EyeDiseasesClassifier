package eyepred.eyediseasesclassifier; // Ganti dengan package Anda

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class App extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("MainView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 1600, 850); // Ukuran window diperbesar
        
        // 1. Judul aplikasi pada window
        stage.setTitle("Interactive Image Classifier (VGG16)");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}