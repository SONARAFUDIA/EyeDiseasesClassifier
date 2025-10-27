package eyepred.eyediseasesclassifier;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

/**
 * JavaFX App (Main Entry Point)
 * Nama kelas ini "App" untuk mencocokkan <mainClass> di pom.xml
 */
public class App extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        // Lokasi FXML disesuaikan dengan package
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource("MainView.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 900, 750); // Ukuran window awal
        stage.setTitle("Eye Disease Classifier (STKI Project)");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}