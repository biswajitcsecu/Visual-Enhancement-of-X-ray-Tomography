#include <cmath>
#include <iostream>
#include <stdlib.h>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "arrayfire.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace af;


int main(int argc, const char** argv)
{
    try {
        system ("CLS");

        const string filename = "src.jpg";

        Mat src = imread(filename, IMREAD_GRAYSCALE);
        Mat mask;
        cv::Canny(src, mask, 100, 200, 3);

        Mat dst_cpu;
        cv::cvtColor(mask, dst_cpu, COLOR_GRAY2BGR);
        Mat dst_gpu = dst_cpu.clone();

        /* pixel manipulation*/
        Mat grc, srctem, grccom;
        size_t nrows=0, ncols=0, nchn=0;
        nrows=src.rows;
        ncols=src.cols;
        nchn=src.channels();
        std::cout<<"rows, cols and chanels"<<'\t'<<nrows<<'\t'<<ncols
                <<'\t'<<nchn<<std::endl;

        src.copyTo(grc);
        grc.convertTo(srctem,CV_32FC1);

        // number of lines
        int nl= srctem.rows;
        // total number of elements per line
        int nc= srctem.cols * srctem.channels();
        //fuzzification
        int div=255;
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* data= srctem.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                data[i]= cv::saturate_cast<float>((div-data[i])/(div));
            }
        }

        //fuzz entropy
        double sum=0, ent=0;
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* data= srctem.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                ent= cv::saturate_cast<float>(-(sum+data[i]*log(data[i])));
            }
        }
        std::cout<<"Entropy:\t"<<ent<<std::endl;

        //membershipgrade modification
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* data= srctem.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                if(data[i]>ent)
                    data[i]= cv::saturate_cast<float>((data[i]*data[i]));
                else
                    data[i]= cv::saturate_cast<float>(2*(data[i]*data[i])-1);
            }
        }

        //fuzz intensity relaxtation
        srctem.copyTo(grccom);
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* datacom= grccom.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                datacom[i]= cv::saturate_cast<float>(abs(1-datacom[i]));

            }
        }

        for(int j=0;j<nl;j++){
            // get the address of row j
            float* datacom= grccom.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                if(datacom[i]>2*ent){
                datacom[i]=(1-datacom[i])/(1+ent*tanh(datacom[i]));
                datacom[i]=abs((datacom[i]/(ent*(1+datacom[i]))));
                datacom[i]= cv::saturate_cast<float>(datacom[i]);
                // std::cout<<datacom[i];
                }
                else{
                datacom[i]=(2*datacom[i])/(1-ent*tanh(datacom[i]));
                datacom[i]=abs((datacom[i]/(ent*(1-datacom[i]))));
                datacom[i]= cv::saturate_cast<float>(datacom[i]);
                // std::cout<<datacom[i];
                }


            }

        }
        double fac=300000;
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* datacom= grccom.ptr<float>(j);
            float* data= srctem.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                data[i]=data[i]+abs(max(datacom[i],data[i])-min(datacom[i],data[i]))
                        *(ent/fac)+(ent*(min(datacom[i],data[i]))+
                                    ((ent/fac)*(datacom[i]-data[i])/2));
                data[i]= cv::saturate_cast<float>(data[i]);
                //std::cout<<datacom[i];
            }
        }

        //defuzz
        for(int j=0;j<nl;j++){
            // get the address of row j
            float* data= srctem.ptr<float>(j);
            for (int i=0; i<nc; i++) {
                data[i]= cv::saturate_cast<float>(div-(div*data[i]));
            }
        }













        //convert to image plane
        srctem.convertTo(srctem,CV_8UC1);


        /*hough line detection*/
        vector<Vec4i> lines_cpu;
        {
            const int64 start = getTickCount();

            cv::HoughLinesP(mask, lines_cpu, 1, CV_PI / 180, 50, 60, 5);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "CPU Time : " << timeSec * 1000 << " ms" << endl;
            cout << "CPU Found : " << lines_cpu.size() << endl;
        }

        for (size_t i = 0; i < lines_cpu.size(); ++i)
        {
            Vec4i l = lines_cpu[i];
            line(dst_cpu, Point(l[0], l[1]), Point(l[2], l[3]),
                    Scalar(0, 0, 255), 3, LINE_AA);
        }

        GpuMat d_src(mask);
        GpuMat d_lines;
        {
            const int64 start = getTickCount();

            Ptr<cuda::HoughSegmentDetector> hough =
                    cuda::createHoughSegmentDetector(1.0f, (float)
                                                     (CV_PI / 180.0f), 50, 5);

            hough->detect(d_src, d_lines);

            const double timeSec = (getTickCount() - start) / getTickFrequency();
            cout << "GPU Time : " << timeSec * 1000 << " ms" << endl;
            cout << "GPU Found : " << d_lines.cols << endl;
        }

        vector<Vec4i> lines_gpu;
        if (!d_lines.empty())
        {
            lines_gpu.resize(d_lines.cols);
            Mat h_lines(1, d_lines.cols, CV_32SC4, &lines_gpu[0]);
            d_lines.download(h_lines);
        }

        for (size_t i = 0; i < lines_gpu.size(); ++i)
        {
            Vec4i l = lines_gpu[i];
            line(dst_gpu, Point(l[0], l[1]), Point(l[2], l[3]),
                    Scalar(0, 0, 255), 3, LINE_AA);
        }

        cv::namedWindow( "Enhanced", WINDOW_AUTOSIZE );
        cv::imshow( "Enhanced", srctem );

        cv::namedWindow( "source", WINDOW_AUTOSIZE );
        imshow("source", src);

        //cv::namedWindow( "detected lines [CPU]", WINDOW_AUTOSIZE );
        //imshow("detected lines [CPU]", dst_cpu);

        //cv::namedWindow( "detected lines [GPU]", WINDOW_AUTOSIZE );
        //imshow("detected lines [GPU]", dst_gpu);

        waitKey();
        destroyAllWindows();

    } catch (std::exception &excp) {
        std::cerr<<"Error found"<<excp.what()<<std::endl;
    }
    return EXIT_SUCCESS;
}
