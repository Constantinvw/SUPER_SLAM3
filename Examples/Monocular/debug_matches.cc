#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "ORBextractor.h"

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

// ---------------------- Parameter-Struktur ----------------------

struct DebugParams
{
    double clahe_clip_limit = 3.0;   // CLAHE Kontrastlimit
    double distance_factor  = 0.7;   // Faktor für "good matches" (distance < factor * maxDist)
    bool   use_ransac       = true;  // RANSAC zur Inlier/Outlier-Trennung
    double ransac_threshold = 6.0;   // RANSAC-Pixel-Threshold
    double ransac_conf      = 0.99;  // RANSAC-Confidence
};

DebugParams loadParams(const std::string& configPath)
{
    DebugParams p;

    FileStorage fs(configPath, FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr << "WARNUNG: Konnte Config " << configPath
             << " nicht öffnen. Verwende Default-Parameter." << endl;
        return p;
    }

    // Nur lesen, falls vorhanden – sonst Default behalten
    if (!fs["clahe_clip_limit"].empty())
        fs["clahe_clip_limit"] >> p.clahe_clip_limit;

    if (!fs["distance_factor"].empty())
        fs["distance_factor"] >> p.distance_factor;

    if (!fs["use_ransac"].empty())
    {
        int use_r;
        fs["use_ransac"] >> use_r;
        p.use_ransac = (use_r != 0);
    }

    if (!fs["ransac_threshold"].empty())
        fs["ransac_threshold"] >> p.ransac_threshold;

    if (!fs["ransac_confidence"].empty())
        fs["ransac_confidence"] >> p.ransac_conf;

    return p;
}

// ---------------------- Canvas & Zeichnen ----------------------

static Mat makeMatchCanvas(const Mat& img1, const Mat& img2)
{
    int h = std::max(img1.rows, img2.rows);
    int w = img1.cols + img2.cols;

    Mat out(h, w, CV_8UC3, Scalar(0, 0, 0));

    Mat left  = out(Rect(0,         0, img1.cols, img1.rows));
    Mat right = out(Rect(img1.cols, 0, img2.cols, img2.rows));

    if (img1.channels() == 1)
        cvtColor(img1, left, COLOR_GRAY2BGR);
    else
        img1.copyTo(left);

    if (img2.channels() == 1)
        cvtColor(img2, right, COLOR_GRAY2BGR);
    else
        img2.copyTo(right);

    return out;
}

static void drawThickMatchesOnCanvas(Mat& canvas,
                                     const Mat& img1,
                                     const vector<KeyPoint>& kps1,
                                     const Mat& img2,
                                     const vector<KeyPoint>& kps2,
                                     const vector<DMatch>& matches,
                                     const Scalar& color)
{
    const int line_thickness = 2;
    const int point_radius   = 5;

    for (const auto& m : matches)
    {
        Point2f p1 = kps1[m.queryIdx].pt;
        Point2f p2 = kps2[m.trainIdx].pt + Point2f((float)img1.cols, 0.f);

        line(canvas, p1, p2, color, line_thickness, LINE_AA);
        circle(canvas, p1, point_radius, color, -1, LINE_AA);
        circle(canvas, p2, point_radius, color, -1, LINE_AA);
    }
}

// ---------------------- main ----------------------

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cerr << "Usage: ./debug_matches img1.png img2.png [config.yaml]" << endl;
        return -1;
    }

    string img1_path = argv[1];
    string img2_path = argv[2];

    // 1) Parameter laden (optional: drittes Argument)
    DebugParams params;
    if (argc >= 4)
    {
        string config_path = argv[3];
        params = loadParams(config_path);
        cout << "Config geladen aus: " << config_path << endl;
    }
    else
    {
        cout << "Keine Config angegeben, verwende Default-Parameter." << endl;
    }

    // 2) Bilder laden
    Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
    Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty())
    {
        cerr << "Konnte Bilder nicht laden: " << img1_path
             << " oder " << img2_path << endl;
        return -1;
    }

    // // 3) CLAHE mit Parameter
    // Ptr<CLAHE> clahe = createCLAHE();
    // clahe->setClipLimit(params.clahe_clip_limit);
    // clahe->apply(img1, img1);
    // clahe->apply(img2, img2);

    // 4) ORBextractor (UFEN-SuperPoint)
    int   nfeatures   = 1000;
    float scaleFactor = 1.2f;
    int   nlevels     = 1;
    int   iniThFAST   = 20;
    int   minThFAST   = 7;

    ORBextractor extractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);

    vector<KeyPoint> kps1, kps2;
    Mat desc1, desc2;
    vector<int> vLappingDummy;

    extractor(img1, Mat(), kps1, desc1, vLappingDummy);
    extractor(img2, Mat(), kps2, desc2, vLappingDummy);

    cout << "Bild 1: " << kps1.size() << " Keypoints, "
         << "Bild 2: " << kps2.size() << " Keypoints" << endl;

    if (kps1.empty() || kps2.empty())
    {
        cerr << "Zu wenige Keypoints, Abbruch." << endl;
        return -1;
    }

    // 5) Deskriptor-Matching (wie deine erste Version)
    BFMatcher matcher(NORM_HAMMING, true);  // crossCheck = true
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    if (matches.empty())
    {
        cerr << "Keine Matches gefunden." << endl;
        return -1;
    }

    // 6) Distanz-Filter mit Parameter distance_factor
    double maxDist = 0.0;
    for (const auto &m : matches)
        maxDist = max(maxDist, (double)m.distance);

    vector<DMatch> good_matches;
    for (const auto &m : matches)
    {
        if (m.distance < params.distance_factor * maxDist)
            good_matches.push_back(m);
    }

    cout << "Matches total: " << matches.size()
         << ", gute Matches: " << good_matches.size() << endl;

    // 7) Canvas vorbereiten
    Mat out = makeMatchCanvas(img1, img2);

    // Fall 1: Kein RANSAC -> alles grün, nur Distanz-basiert
    if (!params.use_ransac)
    {
        drawThickMatchesOnCanvas(out, img1, kps1, img2, kps2,
                                 good_matches, Scalar(0, 255, 0));
    }
    else
    {
        if (good_matches.size() < 8)
        {
            cerr << "Zu wenige gute Matches fuer F-Matrix, zeichne nur gruen." << endl;
            drawThickMatchesOnCanvas(out, img1, kps1, img2, kps2,
                                     good_matches, Scalar(0, 255, 0));
        }
        else
        {
            // 8) RANSAC-Inlier/Outlier
            vector<Point2f> pts1, pts2;
            pts1.reserve(good_matches.size());
            pts2.reserve(good_matches.size());

            for (const auto& m : good_matches)
            {
                pts1.push_back(kps1[m.queryIdx].pt);
                pts2.push_back(kps2[m.trainIdx].pt);
            }

            vector<uchar> inlierMask;
            Mat F = findFundamentalMat(
                pts1,
                pts2,
                FM_RANSAC,
                params.ransac_threshold,   // aus YAML
                params.ransac_conf,        // aus YAML
                inlierMask
            );

            if (F.empty() || inlierMask.size() != good_matches.size())
            {
                cerr << "FundamentalMat fehlgeschlagen, zeichne alle gruen." << endl;
                drawThickMatchesOnCanvas(out, img1, kps1, img2, kps2,
                                         good_matches, Scalar(0, 255, 0));
            }
            else
            {
                vector<DMatch> inlierMatches, outlierMatches;
                inlierMatches.reserve(good_matches.size());
                outlierMatches.reserve(good_matches.size());

                for (size_t i = 0; i < good_matches.size(); ++i)
                {
                    if (inlierMask[i])
                        inlierMatches.push_back(good_matches[i]);
                    else
                        outlierMatches.push_back(good_matches[i]);
                }

                cout << "RANSAC Inlier: " << inlierMatches.size()
                     << ", Outlier: " << outlierMatches.size() << endl;

                // Outlier: rot
                drawThickMatchesOnCanvas(out, img1, kps1, img2, kps2,
                                         outlierMatches, Scalar(0, 0, 255));

                // Inlier: gruen
                drawThickMatchesOnCanvas(out, img1, kps1, img2, kps2,
                                         inlierMatches, Scalar(0, 255, 0));
            }
        }
    }

    string out_name = "matches_debug.png";
    imwrite(out_name, out);
    cout << "Ergebnis gespeichert als: " << out_name << endl;

    return 0;
}