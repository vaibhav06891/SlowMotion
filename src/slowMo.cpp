#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


#define CLAMP(x,min,max) (  ((x) < (min)) ? (min) : ( ((x) > (max)) ? (max) : (x) )  )

int main(int argc, char** argv)
{
	Mat frame,prevframe;
	Mat prevgray, gray;
	Mat fflow,bflow;
	string videoName = argv[1];  // the video filename should be given as the arguement while executing the program
	VideoCapture capture(videoName);


		//--------------------initialize video writer object---------------------------------------
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	int fps = (int)capture.get(CV_CAP_PROP_FPS);


	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
	VideoWriter oVideoWriter ("SlowVideo1.avi", CV_FOURCC('P','I','M','1'), fps, frameSize, true); //initialize the VideoWriter object

	if ( !oVideoWriter.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
	{	cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}
	capture >> frame;

	Mat flowf(frame.rows,frame.cols ,CV_8UC3);   // the forward co-ordinates for interpolation
	flowf.setTo(Scalar(255,255,255));
	Mat flowb(frame.rows,frame.cols ,CV_8UC3);   // the backward co-ordinates for interpolation
	flowb.setTo(Scalar(255,255,255));
	Mat final(frame.rows,frame.cols ,CV_8UC3);

	int fx,fy,bx,by;

	for(;;)
	{	capture >> frame;
		if(!frame.data)
			break;

		if(prevframe.data)
		{
			cvtColor(prevframe,prevgray,COLOR_BGR2GRAY);  // Convert to gray space for optical flow calculation
			cvtColor(frame, gray, COLOR_BGR2GRAY);
			calcOpticalFlowFarneback(prevgray, gray, fflow, 0.5, 3, 15, 3, 3, 1.2, 0);  // forward optical flow
			calcOpticalFlowFarneback(gray, prevgray, bflow, 0.5, 3, 15, 3, 3, 1.2, 0);   //backward optical flow

			for (int t=0;t<4;t++)     // interpolating 20 frames from two given frames at different time locations
			{
				// less than original rows and columns scanned to avoid having co-ordinates outside the frame size
				for (int y=0; y<frame.rows; y++) //column scan
				{
					for (int x=0; x<frame.cols; x++) //row scan
					{
						const Point2f fxy = fflow.at<Point2f>(y,x);
						fy = CLAMP(y+fxy.y*0.25*t,0,780);
						fx = CLAMP(x+fxy.x*0.25*t,0,1280);

						flowf.at<Vec3b>(fy,fx) = prevframe.at<Vec3b>(y,x);

						const Point2f bxy = bflow.at<Point2f>(y,x);
						by = CLAMP(y+bxy.y*(1-0.25*t),0,780);
						bx = CLAMP(x+bxy.x*(1-0.25*t),0,1280);
						flowb.at<Vec3b>(by,bx) = frame.at<Vec3b>(y,x);					}
				}
				final = flowf*(1-0.25*t) + flowb*0.25*t;  //combination of frwd and bckward martrix
				cv::medianBlur(final,final,3);
				oVideoWriter.write(final);
				}
		}

		swap(frame,prevframe);

	}
	return 0;
}
