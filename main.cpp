#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h> // Pour GetSystemMetrics et gestion du Hook
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace dnn;
using namespace std;

// Charger la DLL de contrôle de la souris
typedef void (*MoveMouseFunc)(int, int);
typedef void (*ClickMouseFunc)();

HINSTANCE hinstLib;
MoveMouseFunc move_mouse;
ClickMouseFunc click_mouse;

bool mouse_control_active = false; // Activation/désactivation de la souris

// Hook clavier global
LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION) {
        KBDLLHOOKSTRUCT* pKeyboard = (KBDLLHOOKSTRUCT*)lParam;
        if (wParam == WM_KEYDOWN) {
            if (pKeyboard->vkCode == 'T') {  // Touche "T" détectée
                mouse_control_active = !mouse_control_active;
                cout << "Contrôle de la souris " << (mouse_control_active ? "ACTIVÉ ✅" : "DÉSACTIVÉ ❌") << endl;
            }
            if (pKeyboard->vkCode == 'Q') {  // Touche "Q" détectée
                cout << "Fermeture du programme..." << endl;
                exit(0);
            }
        }
    }
    return CallNextHookEx(NULL, nCode, wParam, lParam);
}

// Fonction pour capturer l'écran
Mat captureScreen() {
    HDC hdcScreen = GetDC(NULL);
    HDC hdcMem = CreateCompatibleDC(hdcScreen);
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    HBITMAP hbmScreen = CreateCompatibleBitmap(hdcScreen, screenWidth, screenHeight);
    SelectObject(hdcMem, hbmScreen);
    BitBlt(hdcMem, 0, 0, screenWidth, screenHeight, hdcScreen, 0, 0, SRCCOPY);

    Mat matScreen(screenHeight, screenWidth, CV_8UC4);
    GetBitmapBits(hbmScreen, screenWidth * screenHeight * 4, matScreen.data);

    DeleteObject(hbmScreen);
    DeleteDC(hdcMem);
    ReleaseDC(NULL, hdcScreen);

    cvtColor(matScreen, matScreen, COLOR_BGRA2BGR); // Convertir en format RGB
    return matScreen;
}

// Fonction pour calculer dx/dy pour déplacer la souris
pair<int, int> calculate_dx_dy(int center_x, int center_y, int screen_width, int screen_height) {
    int dx = center_x - (screen_width / 2);
    int dy = center_y - (screen_height / 2);
    dx = max(min(dx, 250), -250);
    dy = max(min(dy, 250), -250);
    return {dx, dy};
}

int main() {
    // Charger le modèle YOLOv8
    string model_path = "yolov8.onnx";
    Net net = readNetFromONNX(model_path);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    if (net.empty()) {
        cerr << "❌ ERREUR : Impossible de charger le modèle YOLOv8 ONNX !" << endl;
        return -1;
    } else {
        cout << "✅ Modèle ONNX chargé avec succès." << endl;
    }

    // Charger la DLL de contrôle de la souris
    hinstLib = LoadLibrary(TEXT("own.dll"));
    if (!hinstLib) {
        cerr << "Erreur : Impossible de charger la DLL de la souris !" << endl;
        return -1;
    }
    move_mouse = (MoveMouseFunc)GetProcAddress(hinstLib, "move_mouse");
    click_mouse = (ClickMouseFunc)GetProcAddress(hinstLib, "click_mouse");

    if (!move_mouse || !click_mouse) {
        cerr << "Erreur : Impossible de récupérer les fonctions de la DLL." << endl;
        return -1;
    }
    cout << "DLL de la souris chargée avec succès." << endl;

    // Dimensions de l'écran
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // Activer le Hook Clavier Global
    HHOOK keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, LowLevelKeyboardProc, GetModuleHandle(NULL), 0);
    MSG msg;

    while (true) {
        Mat frame = captureScreen();

        if (frame.empty()) {
            cerr << "Erreur : Frame capturée vide !" << endl;
            continue;
        }

        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);
        cout << "📸 Taille frame avant traitement : " << frame.size() << endl;
        cout << "🔄 Transformation en blob..." << endl;
        cout << "📊 Taille du blob : " << blob.size << endl;

        // Exécuter l'inférence
        Mat output = net.forward();
        cout << "📡 Inférence YOLOv8 terminée, taille de sortie : " << output.size << endl;
        cout << "📝 Vérification de la structure des données YOLOv8 ONNX..." << endl;
        cout << "📌 Nombre total d'éléments : " << output.total() << endl;
        cout << "📏 Dimensions : " << output.size << endl;


        // Parcourir les détections
        cout << "📝 Premiers éléments de la sortie YOLOv8 :" << endl;
        for (int i = 0; i < output.rows; i++) {
            float confidence = output.at<float>(i, 4);
            if (confidence > 0.5) {
                cout << "🎯 Détection #" << i << " - Confiance : " << confidence << endl;
                int x1 = round(output.at<float>(i, 0) * screenWidth) - 50;
                int y1 = round(output.at<float>(i, 1) * screenHeight) - 50;
                int x2 = round(output.at<float>(i, 0) * screenWidth) + 50;
                int y2 = round(output.at<float>(i, 1) * screenHeight) + 50;

                int x_center = round((x1 + x2) / 2);
                int y_center = round((y1 + y2) / 2);

                cout << "🎯 Détection #" << i << " - Confiance : " << confidence 
                     << " | X: " << x_center << " | Y: " << y_center << endl;

                if (mouse_control_active) {
                    move_mouse(x_center - screenWidth / 2, y_center - screenHeight / 2);
                    click_mouse();
                }
            }
        }

        imshow("YOLOv8 C++ Detection", frame);
        waitKey(1); // Nécessaire pour rafraîchir l'image OpenCV

        // Écouter les événements clavier (hook)
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    UnhookWindowsHookEx(keyboardHook); // Libérer le hook avant de quitter
    FreeLibrary(hinstLib);
    return 0;
}
