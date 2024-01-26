#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

//CALIBRATION FUNCTION

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2, MEASURING = 3 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
			cameraMatrix, distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err * err / n);
		totalErr += err * err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
	corners.resize(0);

	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float(j * squareSize),
					float(i * squareSize), 0));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
					float(i * squareSize), 0));
		break;

	default:
		CV_Error(Error::StsBadArg, "Unknown pattern type\n");
	}
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
	Size imageSize, Size boardSize, Pattern patternType,
	float squareSize, float aspectRatio,
	float grid_width, bool release_object,
	int flags, Mat& cameraMatrix, Mat& distCoeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs,
	vector<Point3f>& newObjPoints,
	double& totalAvgErr)
{
	if (flags & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);
	objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
	newObjPoints = objectPoints[0];

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms;
	int iFixedPoint = -1;
	if (release_object)
		iFixedPoint = boardSize.width - 1;
	rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
		cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
		flags | CALIB_USE_LU);
	printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	if (release_object) {
		cout << "New board corners: " << endl;
		cout << newObjPoints[0] << endl;
		cout << newObjPoints[boardSize.width - 1] << endl;
		cout << newObjPoints[boardSize.width * (boardSize.height - 1)] << endl;
		cout << newObjPoints.back() << endl;
	}

	objectPoints.clear();
	objectPoints.resize(imagePoints.size(), newObjPoints);
	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

static bool loadCameraParams(const string& filename, Size& imageSize, Size& boardSize, Mat& cameraMatrix, Mat& distCoeffs, float& squareSize, double& totalAvgErr) {
	FileStorage fs(filename, FileStorage::READ);

	imageSize.width = fs["image_width"];
	imageSize.height = fs["image_height"];
	boardSize.width = fs["board_width"];
	boardSize.height = fs["board_height"];
	fs["square_size"] >> squareSize;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;

	totalAvgErr = fs["avg_reprojection_error"];
	return (fs.isOpened());
}

static void saveCameraParams(const string& filename,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Point3f>& newObjPoints,
	double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t tt;
	time(&tt);
	struct tm* t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags & CALIB_FIX_ASPECT_RATIO)
		fs << "aspectRatio" << aspectRatio;

	if (flags != 0)
	{
		snprintf(buf, sizeof(buf), "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
		//cvWriteComment( *fs, buf, 0 );
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		//cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}

	if (!newObjPoints.empty())
	{
		fs << "grid_points" << newObjPoints;
	}
}

static void createPlane(vector<cv::Point2f>& projectedPlanes,
	std::vector<Point>& Xplane, std::vector<Point>& Yplane, std::vector<Point>& Zplane) {

	Xplane.push_back({ Point(projectedPlanes.at(0)) });
	Xplane.push_back({ Point(projectedPlanes.at(1)) });
	Xplane.push_back({ Point(projectedPlanes.at(4)) });
	Xplane.push_back({ Point(projectedPlanes.at(3)) });

	Yplane.push_back({ Point(projectedPlanes.at(0)) });
	Yplane.push_back({ Point(projectedPlanes.at(2)) });
	Yplane.push_back({ Point(projectedPlanes.at(5)) });
	Yplane.push_back({ Point(projectedPlanes.at(3)) });

	Zplane.push_back({ Point(projectedPlanes.at(0)) });
	Zplane.push_back({ Point(projectedPlanes.at(1)) });
	Zplane.push_back({ Point(projectedPlanes.at(6)) });
	Zplane.push_back({ Point(projectedPlanes.at(2)) });

}

static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	size_t dir_pos = filename.rfind('/');
	if (dir_pos == string::npos)
		dir_pos = filename.rfind('\\');
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		string fname = (string)*it;
		if (dir_pos != string::npos)
		{
			string fpath = samples::findFile(filename.substr(0, dir_pos + 1) + fname, false);
			if (fpath.empty())
			{
				fpath = samples::findFile(fname);
			}
			fname = fpath;
		}
		else
		{
			fname = samples::findFile(fname);
		}
		l.push_back(fname);
	}
	return true;
}

static bool runAndSave(const string& outputFilename,
	const vector<vector<Point2f> >& imagePoints,
	Size imageSize, Size boardSize, Pattern patternType, float squareSize,
	float grid_width, bool release_object,
	float aspectRatio, int flags, Mat& cameraMatrix,
	Mat& distCoeffs, bool writeExtrinsics, bool writePoints, bool writeGrid,
	double& totalAvgErr, vector<float>& reprojErrs)
{
	vector<Mat> rvecs, tvecs;
	vector<Point3f> newObjPoints;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
		aspectRatio, grid_width, release_object, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, newObjPoints, totalAvgErr);
	printf("%s. avg reprojection error = %.7f\n",
		ok ? "Calibration succeeded" : "Calibration failed",
		totalAvgErr);

	if (ok)
		saveCameraParams(outputFilename, imageSize,
			boardSize, squareSize, aspectRatio,
			flags, cameraMatrix, distCoeffs,
			writeExtrinsics ? rvecs : vector<Mat>(),
			writeExtrinsics ? tvecs : vector<Mat>(),
			writeExtrinsics ? reprojErrs : vector<float>(),
			writePoints ? imagePoints : vector<vector<Point2f> >(),
			writeGrid ? newObjPoints : vector<Point3f>(),
			totalAvgErr);
	return ok;
}
//END CALIBRATION FUNCTION