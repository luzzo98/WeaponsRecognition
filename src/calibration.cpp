#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace cv::dnn;

#include "Calibration.h"
#include "net_utils.h"

const char* usage =
" \nexample command line for calibration from a live feed.\n"
"   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe\n"
" \n"
" example command line for calibration from a list of stored images:\n"
"   imagelist_creator image_list.xml *.png\n"
"   calibration -w=4 -h=5 -s=0.025 -o=camera.yml -op -oe image_list.xml\n"
" where image_list.xml is the standard OpenCV XML/YAML\n"
" use imagelist_creator to create the xml or yaml list\n"
" file consisting of the list of strings, e.g.:\n"
" \n"
"<?xml version=\"1.0\"?>\n"
"<opencv_storage>\n"
"<images>\n"
"view000.png\n"
"view001.png\n"
"<!-- view002.png -->\n"
"view003.png\n"
"view010.png\n"
"one_extra_view.jpg\n"
"</images>\n"
"</opencv_storage>\n";

const char* liveCaptureHelp =
"When the live video from camera is used as input, the following hot-keys may be used:\n"
"  <ESC>, 'q' - quit the program\n"
"  'g' - start capturing images\n"
"  'u' - switch undistortion on/off\n";

static void help(char** argv)
{
    printf("This is a camera calibration sample.\n"
        "Usage: %s\n"
        "     -w=<board_width>         # the number of inner corners per one of board dimension\n"
        "     -h=<board_height>        # the number of inner corners per another board dimension\n"
        "     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
        "     [-n=<number_of_frames>]  # the number of frames to use for calibration\n"
        "                              # (if not specified, it will be set to the number\n"
        "                              #  of board views actually available)\n"
        "     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
        "                              # (used only for video capturing)\n"
        "     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
        "     [-o=<out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
        "     [-op]                    # write detected feature points\n"
        "     [-oe]                    # write extrinsic parameters\n"
        "     [-oo]                    # write refined 3D object points\n"
        "     [-zt]                    # assume zero tangential distortion\n"
        "     [-a=<aspectRatio>]       # fix aspect ratio (fx/fy)\n"
        "     [-p]                     # fix the principal point at the center\n"
        "     [-v]                     # flip the captured images around the horizontal axis\n"
        "     [-V]                     # use a video file, and not an image list, uses\n"
        "                              # [input_data] string for the video file name\n"
        "     [-su]                    # show undistorted images after calibration\n"
        "     [-ws=<number_of_pixel>]  # half of search window for cornerSubPix (11 by default)\n"
        "     [-fx=<X focal length>]   # focal length in X-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-fy=<Y focal length>]   # focal length in Y-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-cx=<X center point>]   # camera center point in X-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-cy=<Y center point>]   # camera center point in Y-dir as an initial intrinsic guess (if this flag is used, fx, fy, cx, cy must be set)\n"
        "     [-imshow-scale           # image resize scaling factor when displaying the results (must be >= 1)\n"
        "     [-enable-k3=<0/1>        # to enable (1) or disable (0) K3 coefficient for the distortion model\n"
        "     [-dt=<distance>]         # actual distance between top-left and top-right corners of\n"
        "                              # the calibration grid. If this parameter is specified, a more\n"
        "                              # accurate calibration method will be used which may be better\n"
        "                              # with inaccurate, roughly planar target.\n"
        "     [input_data]             # input data, one of the following:\n"
        "                              #  - text file with a list of the images of the board\n"
        "                              #    the text file can be generated with imagelist_creator\n"
        "                              #  - name of video file with a video of the board\n"
        "                              # if input_data not specified, a live view from the camera is used\n"
        "\n", argv[0]);
    printf("\n%s", usage);
    printf("\n%s", liveCaptureHelp);
}

// Variabili aggiunte per calcolare gli errori della calibrazione
vector<float> reprojErrs;
double totalAvgErr = 0;

bool clicked = false;
Point pClicked;
Point pCurPoint;
bool bPaused = false;
Mat point1, point2;
bool pointsSetted;
void onMouseEvent(int evt, int x, int y, int flags, void*) {
    pCurPoint = Point(x, y);
    if (evt == cv::EVENT_LBUTTONDOWN) {
        if (!clicked) {
            point1 = (Mat_<double>(3, 1) << x, y, 1);
        }
        else {
            pointsSetted = true;
            point2 = (Mat_<double>(3, 1) << x, y, 1);
        }
        clicked = true;
    }
}

//################################################################ MAIN ################################################################
int main(int argc, char** argv)
{
    Size boardSize, imageSize;
    float squareSize, aspectRatio = 1;
    Mat cameraMatrix, distCoeffs;
    string outputFilename;
    string inputFilename = "";

    int i, nframes;
    bool writeExtrinsics, writePoints;
    bool undistortImage = false;
    int flags = 0;
    VideoCapture capture;
    bool flipVertical;
    bool showUndistorted;
    bool videofile;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId = 1;
    vector<vector<Point2f> > imagePoints;
    vector<string> imageList;
    Pattern pattern = CHESSBOARD;

    cv::CommandLineParser parser(argc, argv,
        "{help ||}{w||}{h||}{pt|chessboard|}{n|10|}{d|1000|}{s|1|}{o|out_camera_data.yml|}"
        "{op||}{oe||}{zt||}{a||}{p||}{v||}{V||}{su||}"
        "{oo||}{ws|11|}{dt||}"
        "{fx||}{fy||}{cx||}{cy||}"
        "{imshow-scale|1|}{enable-k3|0|}"
        "{@input_data|0|}"
        "{i|out_camera_data.yml|}"); // to load the configuration file
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    //boardSize.width = parser.get<int>("w");
    //boardSize.height = parser.get<int>("h");
    boardSize.width = 10;
    boardSize.height = 7;
    if (parser.has("pt"))
    {
        string val = parser.get<string>("pt");
        if (val == "circles")
            pattern = CIRCLES_GRID;
        else if (val == "acircles")
            pattern = ASYMMETRIC_CIRCLES_GRID;
        else if (val == "chessboard")
            pattern = CHESSBOARD;
        else
            return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
    }
    squareSize = parser.get<float>("s");
    nframes = parser.get<int>("n");

    delay = parser.get<int>("d");
    writePoints = parser.has("op");
    writeExtrinsics = parser.has("oe");
    bool writeGrid = parser.has("oo");
    if (parser.has("a")) {
        flags |= CALIB_FIX_ASPECT_RATIO;
        aspectRatio = parser.get<float>("a");
    }
    if (parser.has("zt"))
        flags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.has("p"))
        flags |= CALIB_FIX_PRINCIPAL_POINT;
    flipVertical = parser.has("v");
    videofile = parser.has("V");
    if (parser.has("o"))
        outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");
    if (isdigit(parser.get<string>("@input_data")[0]))
        cameraId = parser.get<int>("@input_data");
    else
        inputFilename = parser.get<string>("@input_data");
    int winSize = parser.get<int>("ws");
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (parser.has("fx") && parser.has("fy") && parser.has("cx") && parser.has("cy"))
    {
        cameraMatrix.at<double>(0, 0) = parser.get<double>("fx");
        cameraMatrix.at<double>(0, 2) = parser.get<double>("cx");
        cameraMatrix.at<double>(1, 1) = parser.get<double>("fy");
        cameraMatrix.at<double>(1, 2) = parser.get<double>("cy");
        flags |= CALIB_USE_INTRINSIC_GUESS;
        std::cout << "Use the following camera matrix as an initial guess:\n" << cameraMatrix << std::endl;
    }

    // Carica il file della calibrazione della camera
    if (parser.has("i")) {
        std::string loadFilename = parser.get<string>("i");
        struct stat buffer;
        if (stat(loadFilename.c_str(), &buffer) == 0 && loadCameraParams(loadFilename, imageSize, boardSize, cameraMatrix, distCoeffs, squareSize, totalAvgErr)) {
            mode = CALIBRATED;
        }
    }

    int viewScaleFactor = parser.get<int>("imshow-scale");
    bool useK3 = parser.get<bool>("enable-k3");
    std::cout << "Use K3 distortion coefficient? " << useK3 << std::endl;
    if (!useK3)
    {
        flags |= CALIB_FIX_K3;
    }
    float grid_width = squareSize * (boardSize.width - 1);
    bool release_object = false;
    if (parser.has("dt")) {
        grid_width = parser.get<float>("dt");
        release_object = true;
    }
    if (!parser.check())
    {
        help(argv);
        parser.printErrors();
        return -1;
    }
    if (squareSize <= 0)
        return fprintf(stderr, "Invalid board square width\n"), -1;
    if (nframes <= 3)
        return printf("Invalid number of images\n"), -1;
    if (aspectRatio <= 0)
        return printf("Invalid aspect ratio\n"), -1;
    if (delay <= 0)
        return printf("Invalid delay\n"), -1;
    if (boardSize.width <= 0)
        return fprintf(stderr, "Invalid board width\n"), -1;
    if (boardSize.height <= 0)
        return fprintf(stderr, "Invalid board height\n"), -1;

    if (!inputFilename.empty())
    {
        if (!videofile && readStringList(samples::findFile(inputFilename), imageList))
            mode = CAPTURING;
        else
            capture.open(samples::findFileOrKeep(inputFilename));
    }
    else
        capture.open(cameraId);

    if (!capture.isOpened() && imageList.empty())
        return fprintf(stderr, "Could not initialize video (%d) capture\n", cameraId), -2;

    if (!imageList.empty())
        nframes = (int)imageList.size();

    if (capture.isOpened())
        printf("%s", liveCaptureHelp);

    const char* winName = "Image View";
    namedWindow(winName, 1);
    // Viene impostata gestito il click del mouse
    cv::setMouseCallback(winName, onMouseEvent, 0);

    // Calcolo della posizione della scacchiera
    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], pattern);
    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;

    //objectPoints.resize(imagePoints.size(), objectPoints[0]); // cos'era sta roba?

    char key = (char)waitKey(capture.isOpened() ? 50 : 500);
    int index = 0;
    float max_re = 0;
    bool fineTuning = false;
    Mat lastView;

    //********* PROGETTO *********
    Net net;
    net = readNetFromONNX("weights_net.onnx");

    //################################################################ CICLO INFINITO ################################################################
    for (i = 0;; i++)
    {
        Mat view, viewGray;
        bool blink = false;

        if (capture.isOpened() && mode != MEASURING)
        {
            Mat view0;
            capture >> view0;
            view0.copyTo(view);
            view0.copyTo(lastView);
        }
        else if (i < (int)imageList.size())
            view = imread(imageList[i], 1);

        if (mode == MEASURING) {
            lastView.copyTo(view);
        }

        if (view.empty())
        {
            if (imagePoints.size() > 0)
                runAndSave(outputFilename, imagePoints, imageSize,
                    boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                    flags, cameraMatrix, distCoeffs,
                    writeExtrinsics, writePoints, writeGrid,
                    totalAvgErr, reprojErrs);
            break;
        }

        imageSize = view.size();

        if (flipVertical)
            flip(view, view, 0);

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found;
        switch (pattern)
        {
        case CHESSBOARD:
            found = findChessboardCorners(view, boardSize, pointbuf,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
            break;
        case CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf);
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            found = findCirclesGrid(view, boardSize, pointbuf, CALIB_CB_ASYMMETRIC_GRID);
            break;
        default:
            return fprintf(stderr, "Unknown pattern type\n"), -1;
        }

        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found) cornerSubPix(viewGray, pointbuf, Size(winSize, winSize),
            Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        // Se è stata premuta la barra spaziatrice per scattare la foto
        if (key == ' ' && mode == CAPTURING && found &&
            (!capture.isOpened() || clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC))
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();

            // Crea le cartelle per memorizzare le immagini della calibrazione
            namespace fs = std::filesystem;
            fs::create_directory("data");
            fs::create_directory("data_processed");

            // Se ancora si sta effettuando la calibrazione si aggiunge come nuova immagine
            if (!fineTuning) {
                string fname = cv::format("data/img%04d.png", imagePoints.size());
                string fname2 = cv::format("data_processed/img%04d.png", imagePoints.size());
                imwrite(fname, view);
                drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
                imwrite(fname2, view);
            }
            else { // se si è rimossa l'immagine con più errore, questa viene salvata al suo posto
                string fname = cv::format("data/img%04d.png", index + 1);
                string fname2 = cv::format("data_processed/img%04d.png", index + 1);
                imwrite(fname, view);
                drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
                imwrite(fname2, view);
            }
        }

        if (found && mode!= CALIBRATED)
            drawChessboardCorners(view, boardSize, Mat(pointbuf), found);

        // Cerca la foto con maggiore errore tra quelle usate per la calibrazione
        if (reprojErrs.size() > 0) {
            vector<float>::iterator it;
            it = max_element(reprojErrs.begin(), reprojErrs.end());
            max_re = *it;
            index = std::distance(reprojErrs.begin(), it);
        }

        string msg = mode == CAPTURING ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;

        // aggiunta matrici
        Mat rotVec, R, t, K = cameraMatrix;
        double dist = 0.0;

        if (mode == CALIBRATED || mode == MEASURING) {
            msg = cv::format("Calibrated: re=%.2f max=%.2f[%d]", totalAvgErr, max_re, index);
            fineTuning = false;

            //############################################################ INIZIO CALCOLI //############################################################

            if (found) {
                //ricavo rotVec e t se abbiamo rotazione e traslazione (pose estimation), la matrice dell'omografia tra il piano del mondo reale e quello dell'immagine, una sorta di DLT
                solvePnP(objectPoints[0], pointbuf, K, distCoeffs, rotVec, t);
                Rodrigues(rotVec, R); //per comodità abbiamo gli angoli di rotazione nella notazione di rodriguez (matrice di rotazione in un formato specifico)
                Mat Ext;
                hconcat(R, t, Ext); // mette in Ext la matrice di rototraslazione, gli estrinseci
                Mat P = K * Ext; // trovo la P in questo modo calibra moltiplicando per la matrice della camera, contiene K,R e t

                //per trovare dove proiettare un punto basta fare P*[x,y,z]' , ad esempio per un punto sull'asse X di lunghezza 3
                //Mat Xa = (Mat_<double>(4, 1) << 3 * squareSize, 0, 1);
                //Mat Xi = P * Xa;
                //fine

                Mat Hscene2img;
                //homography (planar z=0), x e y sono le prime due colonne, il punto di applicazione è l'ultima colonna (il punto in alto a sinistra della scacchiera) e il piano in cui le metto è z=0
                hconcat(P(Range(0, 3), Range(0, 2)), P.col(3), Hscene2img);
                Mat Himg2scene = Hscene2img.inv(); // serve per riproiettare i punti dall'immagine al mondo (serve per fare le misure dall'immagine)

                // vanishing point(line) degli assi x,y e z
                Matx31d vx = P.col(0);
                Matx31d vy = P.col(1);
                Matx31d vz = P.col(2);
                Matx31d o = P.col(3);

                // normalizzo con (x/z, y/z, 1) (slide capitolo 11 pag 42)
                vx(0) /= vx(2);
                vx(1) /= vx(2);
                vx(2) = 1.0;
                // coordinate omogenee: (x,y,w) -> X=x/w, Y=y/w ; (X,Y,1)= punto del piano / (X,Y,0) punto all'infinito

                vy(0) /= vy(2);
                vy(1) /= vy(2);
                vy(2) = 1.0;

                vz(0) /= vz(2);
                vz(1) /= vz(2);
                vz(2) = 1.0;

                o(0) /= o(2);
                o(1) /= o(2);
                o(2) = 1.0;

                //circle(view, Point(o(0), o(1)), 3, Scalar(255, 0, 0), -1);

                vector<Point3f> scene_axis_point;
                vector<Point2f> projected_axis_point;
                scene_axis_point.push_back(Point3f(3 * squareSize, 0, 0));
                scene_axis_point.push_back(Point3f(0, 3 * squareSize, 0));
                scene_axis_point.push_back(Point3f(0, 0, -3 * squareSize));
                scene_axis_point.push_back(Point3f(0, 0, 0));

                projectPoints(scene_axis_point, rotVec, t, K, distCoeffs, projected_axis_point);

                //line(view, Point(vx(0), vx(1)), Point(vy(0), vy(1)), Scalar(255, 255, 255), 2); // linea che congiunge i punti di fuga di x e y
                arrowedLine(view, Point2f(o(0), o(1)), projected_axis_point[0], Scalar(255, 0, 0));
                arrowedLine(view, Point2f(o(0), o(1)), projected_axis_point[1], Scalar(0, 255, 0));
                arrowedLine(view, Point2f(o(0), o(1)), projected_axis_point[2], Scalar(0, 0, 255));
                circle(view, projected_axis_point[3], 3, Scalar(255, 0, 0), 3);

                putText(view, "X", projected_axis_point[0], 1, 2, Scalar(255, 0, 0));
                putText(view, "Y", projected_axis_point[1], 1, 2, Scalar(0, 255, 0));
                putText(view, "Z", projected_axis_point[2], 1, 2, Scalar(0, 0, 255));

                //Mat tmp = (Mat_<double>(4, 1) << 1.0, 0.0, 0.0, 0.0);
                //Mat X = P * tmp;
                //X = X.at<double>(0) / X.at<double>(2);
                //cout << objectPoints[0] << endl << R << endl << t << endl << cameraMatrix << endl << distCoeffs;

                // aggiunta 4
                Mat normal = (Mat_<double>(3, 1) << 0, 0, 1);
                Mat normal1 = R * normal;
                Mat origin(3, 1, CV_64F, Scalar(0)); // riempo una matrice 3x1 di scalari 0
                Mat origin1 = R * origin + t;
                dist = normal1.dot(origin1);

                // aggiunta 4.2

                if (key == ' ') {
                    mode = MEASURING;
                    cout << "PAUSA";
                }

                // Himg2scene
                vector<Point2f> misurationPoints;
                if (clicked == true) {

                }
            }

            //############################################################ FINE CALCOLI //############################################################
        }

        if (mode == MEASURING && key == 'f') {
            mode = CALIBRATED;
        }

        if (mode == CAPTURING)
        {
            if (undistortImage)
                msg = cv::format("%d/%d Undist", (int)imagePoints.size(), nframes);
            else
                msg = cv::format("%d/%d", (int)imagePoints.size(), nframes);
        }

        if (found) {
            msg = cv::format("d=%.2f mm (%.2f)", dist, cv::norm(t));
        }

        // aggiunta 4
        if (mode == MEASURING) {
            msg += " [||] ";
        }

        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 3 * baseLine - 10);

        putText(view, msg, textOrigin, 1, 1,
            mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

        if (blink)
            bitwise_not(view, view);

        if (mode == CALIBRATED && undistortImage)
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);
        }
        if (viewScaleFactor > 1)
        {
            Mat viewScale;
            resize(view, viewScale, Size(), 1.0 / viewScaleFactor, 1.0 / viewScaleFactor, INTER_AREA);
            imshow("Image View", viewScale);
        }
        else
        {
            imshow("Image View", view);
        }

        key = (char)waitKey(capture.isOpened() ? 50 : 500);

        // Premendo "A" al termine della calibrazione si rimuove l'immagine con l'errore maggiore
        if ((key == 'a') && mode == CALIBRATED && !imagePoints.empty()) {
            cout << "Max el:" << index << endl;
            imagePoints.erase(imagePoints.begin() + index);
            string path_pro = cv::format("data_processed/img%04d.png", index + 1);
            string path = cv::format("data/img%04d.png", index + 1);
            remove(path.c_str());
            remove(path_pro.c_str());
            fineTuning = true;
            mode = CAPTURING;
        }

        if (key == 27)
            break;

        if (key == 'u' && mode == CALIBRATED)
            undistortImage = !undistortImage;

        if (capture.isOpened() && key == 'g')
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        // *************** PROGETTO ***************
        if (key == 'w' && mode != PROJECT)
        {
            mode = PROJECT;





















            vector<Mat> detections;
            detections = pre_process(view, net);

            Mat temp = view.clone();
            Mat img = post_process(temp, detections);

            // Put efficiency information.
            // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            string label = cv::format("Inference time : %.2f ms", t);
            putText(img, label, Point(20, 420), FONT_FACE, FONT_SCALE, RED);

            imshow("Output", img);
            waitKey(0);




























        }

        if (mode == CAPTURING && imagePoints.size() >= (unsigned)nframes)
        {
            if (runAndSave(outputFilename, imagePoints, imageSize,
                boardSize, pattern, squareSize, grid_width, release_object, aspectRatio,
                flags, cameraMatrix, distCoeffs,
                writeExtrinsics, writePoints, writeGrid, totalAvgErr, reprojErrs))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if (!capture.isOpened())
                break;
        }
    }

    if (!capture.isOpened() && showUndistorted)
    {
        Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);

        for (i = 0; i < (int)imageList.size(); i++)
        {
            view = imread(imageList[i], 1);
            if (view.empty())
                continue;
            remap(view, rview, map1, map2, INTER_LINEAR);
            if (viewScaleFactor > 1)
            {
                Mat rviewScale;
                resize(rview, rviewScale, Size(), 1.0 / viewScaleFactor, 1.0 / viewScaleFactor, INTER_AREA);
                imshow("Image View", rviewScale);
            }
            else
            {
                imshow("Image View", rview);
            }
            char c = (char)waitKey();
            if (c == 27 || c == 'q' || c == 'Q')
                break;
        }
    }

    return 0;
}
