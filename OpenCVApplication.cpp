// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "Functions.h"

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

void testContour()
{
	Mat src;
	Mat dst;

	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		imshow("src", src);

		// Aplicare FTJ gaussian pt. eliminare zgomote
		// http://opencvexamples.blogspot.com/2013/10/applying-gaussian-filter.html
		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat dst = Mat::zeros(src.size(), src.type());

		//de testat pe imaginea cu monede: eight.bmp
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val < 200)
					dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("Binarizare", dst);

		// --------------------------------- Operatii morfologice ----------------------------------
		//structuring element for morpho operations
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(dst, dst, element, Point(-1, -1), 2);
		erode(dst, dst, element, Point(-1, -1), 2);

		imshow("Postprocesare", dst);

		Labeling("Contur - functii din OpenCV", dst, true);

		// --------------------------- Proprietati geometrice simple ---------------------------------

		// Wait until user press some key
		waitKey(0);
	}
}

/////////////////////////////////////////////////PROIECT//////////////////////////////////////////

#define MAX_LUV 256

int histc_u[MAX_LUV];
int histc_v[MAX_LUV];

int histFiltrat_u[MAX_LUV];
int histFiltrat_v[MAX_LUV];

bool trained = false;

void ColorModel_Init() {
	memset(histc_u, 0, sizeof(unsigned int) * MAX_LUV);
	memset(histc_v, 0, sizeof(unsigned int) * MAX_LUV);
	trained = false;
}

///lab 2 calculam media si dev standard pe histograma

void computeMeanDevFromHist(int hist_src[MAX_LUV], int hist_dst[MAX_LUV], double& mean, double& dev) {

	double  M = 0; //nr total de pixeli
	int max = 0;

	for (int i = 0; i < MAX_LUV; i++) {
		M += hist_src[i];
		if (hist_src[i] > max) {
			max = hist_src[i];
		}
	}

		int T = (int)(max * 0.1);

		printf("T= %d\n", T);

		double suma = 0;
		double M_filtrat = 0;

		for (int i = 0; i < MAX_LUV; i++) {
			if (hist_src[i] >= T) {
				suma += i * hist_src[i];
				M_filtrat += hist_src[i];
				hist_dst[i] = hist_src[i];
			}
			else {
				hist_dst[i] = 0;
			}
		}

		mean = suma / M_filtrat; 

		double sum2 = 0;

		for (int i = 0; i < MAX_LUV; i++) {
			if (hist_src[i] >= T) {
				sum2 += pow((double)i - mean, 2) * hist_src[i];
			}
		}

		dev = sqrt(sum2 / M_filtrat);

}

int train_limit = 10;

double med_u = 0, dev_u = 0;
double med_v = 0, dev_v = 0;



////////////////////////////////////////////////////////////////////////////////STATIC

void faceDetectionVJStatic() {

	CascadeClassifier face_cascade;
	if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
		printf("Error loading face cascades!\n");
		return;
	}

	String training_folder = "Images/train/*.jpg";
	std::vector<String> file_list;
	cv::glob(training_folder, file_list, false);

	if (file_list.empty()) {
		printf("ERROR: No image found\n");
		return;
	}

	ColorModel_Init();
	int imgNum = -1;

	for (const String& filename : file_list) {
		Mat frame = imread(filename);
		if (frame.empty()) continue;

		++imgNum;
		Mat frame_copy = frame.clone();
		Mat gray, luv, channels[3];

		cvtColor(frame_copy, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);

		cvtColor(frame_copy, luv, COLOR_BGR2Luv);
		split(luv, channels);
		Mat u = channels[1];
		Mat v = channels[2];

		std::vector<Rect> faces;
		int minFaceSize = 60;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(minFaceSize, minFaceSize));

		if (faces.empty()) {
			printf("No faces detected in %s.\n", filename.c_str());
			continue;
		}

		for (const auto& face_rect : faces) {
			Point center(face_rect.x + face_rect.width * 0.5, face_rect.y + face_rect.height * 0.5);
			float scale_x = 0.6f;
			float scale_y = 0.9f;
			int roi_w = (int)(face_rect.width * scale_x);
			int roi_h = (int)(face_rect.height * scale_y);
			int roi_x = center.x - roi_w / 2;
			int roi_y = center.y - roi_h / 2;

			Rect small_roi(roi_x, roi_y, roi_w, roi_h);
			Rect valid_roi = small_roi & Rect(0, 0, u.cols, u.rows);

			Mat u_roi = u(valid_roi);
			Mat v_roi = v(valid_roi);

			for (int i = 0; i < u_roi.rows; ++i) {
				for (int j = 0; j < u_roi.cols; ++j) {
					uchar u_val = u_roi.at<uchar>(i, j);
					uchar v_val = v_roi.at<uchar>(i, j);
					histc_u[u_val]++;
					histc_v[v_val]++;
				}
			}
			rectangle(frame_copy, valid_roi, Scalar(255, 0, 255), 1);
		}

		char status_msg[100];
		sprintf(status_msg, "ANTRENAMENT... Img %d/%zu", imgNum + 1, file_list.size());
		putText(frame_copy, status_msg, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
		imshow("StaticTrain: Progres", frame_copy);

		showHistogram("U HistCumulativ", histc_u, MAX_LUV, 200, true);
		showHistogram("V HistCumulativ", histc_v, MAX_LUV, 200, true);

		char c = waitKey(0);
		if (c == 27) break;
	}

	computeMeanDevFromHist(histc_u, histFiltrat_u, med_u, dev_u);
	computeMeanDevFromHist(histc_v, histFiltrat_v, med_v, dev_v);

	trained = true;

	printf("\n==================================\n");
	printf("MODEL LUV ANTRENAT STATIC (pe %d imagini)\n", imgNum + 1);
	printf("U: Media = %.2f, Dev = %.2f\n", med_u, dev_u);
	printf("V: Media = %.2f, Dev = %.2f\n", med_v, dev_v);
	printf("==================================\n");

	showHistogram("U HistFiltrata FINALA", histFiltrat_u, MAX_LUV, 200, true);
	showHistogram("V HistFiltrata FINALA", histFiltrat_v, MAX_LUV, 200, true);

	printf("Antrenament GATA. Apasati orice tasta pentru VALIDARE pe imagini de test.\n");
	waitKey(0);

	// --- LOGICA DE VALIDARE  ---

	String test_folder = "Images/test/*.jpg";
	std::vector<String> test_files;
	cv::glob(test_folder, test_files, false);


	for (const String& testFilename : file_list) {
		Mat frame_test = imread(testFilename);
		if (frame_test.empty()) continue;

		Mat resU = frame_test.clone();
		Mat resV = frame_test.clone();
		Mat resBoth = frame_test.clone();

		Mat luv, channels[3];
		cvtColor(frame_test, luv, COLOR_BGR2Luv);
		split(luv, channels);
		Mat u_channel = channels[1];
		Mat v_channel = channels[2];

		Mat gray;
		cvtColor(frame_test, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		std::vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(60, 60));

		float k = 2.5;

		for (const auto& face_rect : faces) {
			
			Mat maskU = Mat::zeros(face_rect.size(), CV_8UC1);
			Mat maskV = Mat::zeros(face_rect.size(), CV_8UC1);
			Mat maskBoth = Mat::zeros(face_rect.size(), CV_8UC1);

			
			for (int i = 0; i < face_rect.height; ++i) {
				for (int j = 0; j < face_rect.width; ++j) {
					uchar u_val = u_channel.at<uchar>(face_rect.y + i, face_rect.x + j);
					uchar v_val = v_channel.at<uchar>(face_rect.y + i, face_rect.x + j);

					bool is_skin_u = (u_val >= (med_u - k * dev_u)) && (u_val <= (med_u + k * dev_u)); 
					bool is_skin_v = (v_val >= (med_v - k * dev_v)) && (v_val <= (med_v + k * dev_v)); 

					if (is_skin_u)
						maskU.at<uchar>(i, j) = 255;
					if (is_skin_v)
						maskV.at<uchar>(i, j) = 255;
					if (is_skin_u && is_skin_v)
						maskBoth.at<uchar>(i, j) = 255;
				}
			}

		//post procesare
			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	
			erode(maskU, maskU, element, Point(-1, -1), 1);
			dilate(maskU, maskU, element, Point(-1, -1), 2);
			erode(maskU, maskU, element, Point(-1, -1), 1);

			erode(maskV, maskV, element, Point(-1, -1), 1);
			dilate(maskV, maskV, element, Point(-1, -1), 2);
			erode(maskV, maskV, element, Point(-1, -1), 1);

			erode(maskBoth, maskBoth, element, Point(-1, -1), 1);
			dilate(maskBoth, maskBoth, element, Point(-1, -1), 2);
			erode(maskBoth, maskBoth, element, Point(-1, -1), 1);

		
			for (int i = 0; i < face_rect.height; ++i) {
				for (int j = 0; j < face_rect.width; ++j) {
					// Colorare pentru U
					if (maskU.at<uchar>(i, j) == 255) {
						resU.at<Vec3b>(face_rect.y + i, face_rect.x + j) = Vec3b(0, 255, 0);
					}
					// Colorare pentru V
					if (maskV.at<uchar>(i, j) == 255) {
						resV.at<Vec3b>(face_rect.y + i, face_rect.x + j) = Vec3b(0, 255, 0);
					}
					// Colorare pentru Combinat (U & V)
					if (maskBoth.at<uchar>(i, j) == 255) {
						resBoth.at<Vec3b>(face_rect.y + i, face_rect.x + j) = Vec3b(0, 255, 0);
					}
				}
			}

		
			rectangle(resU, face_rect, Scalar(255, 0, 255), 2);
			rectangle(resV, face_rect, Scalar(255, 0, 255), 2);

			double ratioBoth = (double)countNonZero(maskBoth) / face_rect.area();
			double percentage = ratioBoth * 100.0;

			bool isValid = (ratioBoth > 0.45);
			Scalar colorStatus = isValid ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
			String label = isValid ? "VALID" : "INVALID";

	
			rectangle(resBoth, face_rect, colorStatus, 2);

		
			char text[100];
			sprintf(text, "%s: %.1f%%", label.c_str(), percentage);

			Point textPos(face_rect.x, face_rect.y - 10);
			if (textPos.y < 20) textPos.y = face_rect.y + 20;

			putText(resBoth, text, textPos, FONT_HERSHEY_SIMPLEX, 0.6, colorStatus, 2);
		
		}

		
		imshow("Validare doar canal U", resU);
		imshow("Validare doar canal V", resV);
		imshow("Validare COMBINATA (U si V)", resBoth);

		if (waitKey(0) == 27) break;
	}

	destroyAllWindows();
}



//////////////////////////////LIVE

void faceDetectionVJLive() {
	CascadeClassifier face_cascade;
	if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
		printf("Error loading face cascades!\n");
		return;
	}

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		printf("ERROR: Could not open video stream!\n");
		return;
	}

	ColorModel_Init();
	int framesCaptured = 0;
	Mat frame;
	double k_factor = 2.5; 

	//ANTRENARE
	printf("MOD ANTRENAMENT: Apasati tasta 'C' pentru a capta un cadru, 'ESC' pentru a anula.\n");

	while (cap.read(frame)) {
		Mat frame_copy = frame.clone();
		Mat gray, luv, channels[3];

		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		cvtColor(frame, luv, COLOR_BGR2Luv);
		split(luv, channels);

		std::vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(60, 60));

		// desenarea roiului pe web-cam
		for (const auto& face_rect : faces) {
			Point center(face_rect.x + face_rect.width * 0.5, face_rect.y + face_rect.height * 0.5);
			float scale_x = 0.6f;
			float scale_y = 0.9f;
			int roi_w = (int)(face_rect.width * scale_x);
			int roi_h = (int)(face_rect.height * scale_y);
			Rect preview_roi = Rect(center.x - roi_w / 2, center.y - roi_h / 2, roi_w, roi_h) & Rect(0, 0, frame.cols, frame.rows);
			rectangle(frame_copy, preview_roi, Scalar(255, 0, 255), 2);
		}

		char status[100];
		sprintf(status, "ANTRENAMENT: %d/%d (Apasati 'C' pt Captura)", framesCaptured, train_limit);
		putText(frame_copy, status, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
		imshow("Live: Antrenament", frame_copy);

		char key = (char)waitKey(1);
		if (key == 27) return; 

		if ((key == 'c' || key == 'C') && !faces.empty()) {
			Rect face_rect = faces[0];
			Point center(face_rect.x + face_rect.width * 0.5, face_rect.y + face_rect.height * 0.5);
			float scale_x = 0.6f;
			float scale_y = 0.9f;
			int roi_w = (int)(face_rect.width * scale_x);
			int roi_h = (int)(face_rect.height * scale_y);
			Rect valid_roi = Rect(center.x - roi_w / 2, center.y - roi_h / 2, roi_w, roi_h) & Rect(0, 0, frame.cols, frame.rows);

			Mat u_roi = channels[1](valid_roi);
			Mat v_roi = channels[2](valid_roi);

			for (int i = 0; i < u_roi.rows; ++i) {
				for (int j = 0; j < u_roi.cols; ++j) {
					histc_u[u_roi.at<uchar>(i, j)]++;
					histc_v[v_roi.at<uchar>(i, j)]++;
				}
			}
			framesCaptured++;
			printf("Cadru %d capturat!\n", framesCaptured);

			if (framesCaptured >= train_limit) break;
		}
	}


	computeMeanDevFromHist(histc_u, histFiltrat_u, med_u, dev_u);
	computeMeanDevFromHist(histc_v, histFiltrat_v, med_v, dev_v);
	trained = true;

	printf("\n--- MODEL LUV ANTRENAT ---\nU: Mean=%.2f, Dev=%.2f | V: Mean=%.2f, Dev=%.2f\n", med_u, dev_u, med_v, dev_v);

	showHistogram("U HistFiltrata FINALA", histFiltrat_u, MAX_LUV, 200, true);
	showHistogram("V HistFiltrata FINALA", histFiltrat_v, MAX_LUV, 200, true);

	printf("Antrenament GATA. Apasati orice tasta in ferestrele de histograma pentru VALIDARE LIVE.\n");
	waitKey(0);
	destroyAllWindows();



	printf("Incepere VALIDARE LIVE... Apasati 'ESC' pentru a inchide.\n");

	while (cap.read(frame)) {
		Mat resU = frame.clone();
		Mat resV = frame.clone();
		Mat resBoth = frame.clone();

		Mat gray, luv, channels[3];
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		cvtColor(frame, luv, COLOR_BGR2Luv);
		split(luv, channels);
		Mat u_channel = channels[1];
		Mat v_channel = channels[2];

		std::vector<Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(60, 60));

		for (const auto& face_rect : faces) {
			Rect safe_rect = face_rect & Rect(0, 0, frame.cols, frame.rows);
			Mat maskU = Mat::zeros(safe_rect.size(), CV_8UC1);
			Mat maskV = Mat::zeros(safe_rect.size(), CV_8UC1);
			Mat maskBoth = Mat::zeros(safe_rect.size(), CV_8UC1);

			for (int i = 0; i < safe_rect.height; ++i) {
				for (int j = 0; j < safe_rect.width; ++j) {
					uchar u_val = u_channel.at<uchar>(safe_rect.y + i, safe_rect.x + j);
					uchar v_val = v_channel.at<uchar>(safe_rect.y + i, safe_rect.x + j);

					bool is_skin_u = (u_val >= (med_u - k_factor * dev_u)) && (u_val <= (med_u + k_factor * dev_u));
					bool is_skin_v = (v_val >= (med_v - k_factor * dev_v)) && (v_val <= (med_v + k_factor * dev_v)); 

					if (is_skin_u) maskU.at<uchar>(i, j) = 255;
					if (is_skin_v) maskV.at<uchar>(i, j) = 255;
					if (is_skin_u && is_skin_v) maskBoth.at<uchar>(i, j) = 255;
				}
			}

			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
			

			erode(maskU, maskU, element, Point(-1, -1), 1);
			dilate(maskU, maskU, element, Point(-1, -1), 2);
			erode(maskU, maskU, element, Point(-1, -1), 1);

			erode(maskV, maskV, element, Point(-1, -1), 1);
			dilate(maskV, maskV, element, Point(-1, -1), 2);
			erode(maskV, maskV, element, Point(-1, -1), 1);

			erode(maskBoth, maskBoth, element, Point(-1, -1), 1);
			dilate(maskBoth, maskBoth, element, Point(-1, -1), 2);
			erode(maskBoth, maskBoth, element, Point(-1, -1), 1);




			for (int i = 0; i < safe_rect.height; ++i) {
				for (int j = 0; j < safe_rect.width; ++j) {
					if (maskU.at<uchar>(i, j) == 255) resU.at<Vec3b>(safe_rect.y + i, safe_rect.x + j) = Vec3b(0, 255, 0);
					if (maskV.at<uchar>(i, j) == 255) resV.at<Vec3b>(safe_rect.y + i, safe_rect.x + j) = Vec3b(0, 255, 0);
					if (maskBoth.at<uchar>(i, j) == 255) resBoth.at<Vec3b>(safe_rect.y + i, safe_rect.x + j) = Vec3b(0, 255, 0);
				}
			}

			rectangle(resU, safe_rect, Scalar(255, 0, 255), 2);
			rectangle(resV, safe_rect, Scalar(255, 0, 255), 2);


			double ratio = (double)countNonZero(maskBoth) / safe_rect.area();
			double percentage = ratio * 100.0;

			bool isValid = (ratio > 0.45);
			Scalar colorStatus = isValid ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
			String label = isValid ? "VALID" : "INVALID";
			rectangle(resBoth, safe_rect, colorStatus, 2);

			
			char infoText[100];
			sprintf(infoText, "%s: %.1f%%", label.c_str(), percentage);

			
			Point textPos(safe_rect.x, safe_rect.y - 10);
			if (textPos.y < 20) textPos.y = safe_rect.y + 20;

			
			putText(resBoth, infoText, textPos, FONT_HERSHEY_SIMPLEX, 0.6, colorStatus, 2);
		}

		imshow("Live: Validare U", resU);
		imshow("Live: Validare V", resV);
		imshow("Live: Validare Combinata", resBoth);

		if (waitKey(1) == 27) break;
	}
	destroyAllWindows();
}

///////////


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - OpenCV labeling\n");
		printf(" 99 - VJ model Luv - STATIC\n");
		printf(" 999 - VJ model Luv - WebCam\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				testContour();
				break;
			case 99:
				faceDetectionVJStatic();
				break;
			case 999:
				faceDetectionVJLive();
				break;
		}
	}
	while (op!=0);
	return 0;
}