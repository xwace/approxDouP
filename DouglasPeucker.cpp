//
// Created by star on 23-3-16.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <stack>
#include "DouglasPeucker.h"
//#define RECUR //用递归的方式实现压缩点

using namespace std;
using namespace cv;

namespace DSP{

    void visualize(const Mat& src, maxDisPair pair, const vector<Point>&con,vector<Point>output){

        Mat out = src.clone().setTo(0);
        cvtColor(out,out,COLOR_GRAY2BGR);
        for (const auto& c: con) {
            out.at<Vec3b>(c) = Vec3b(255,255,255);
        }

        cv::drawMarker(out,get<0>(pair),Scalar(0,0,255),MARKER_CROSS);
        cv::drawMarker(out,get<1>(pair),Scalar(0,0,255),MARKER_CROSS);

        Point lastPt = output[0];
        for (auto o:output) {
            cv::line(out,lastPt,o,Scalar(0,255,0));
            lastPt = o;
        }
        cv::line(out,lastPt,output[0],Scalar(0,255,0));
        cv::namedWindow("out", 2);
        imshow("out", out);
        waitKey();
    }

    /**
     * description: 旋转钳算法,凸包中距离最长的一对点
     * @author oswin
     */
    maxDisPair rotatingCalipers(const vector <cv::Point> &contour) {
        maxDisPair maxPair;
        int q = 1;

        auto xmult = [](Point p1, Point p2, Point p0) {
            return abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y));
        };

        for (int i = 0; i < contour.size() - 1; i++) {

            while (xmult(contour[i + 1], contour[(q + 1) % contour.size()], contour[i]) >
                   xmult(contour[i + 1], contour[q], contour[i])) {
                q = (q + 1) % (int)contour.size();
            }

            int d1 = (int)cv::norm(contour[i] - contour[q]);
            int d2 = (int)cv::norm(contour[i + 1] - contour[q]);

            if (d1 > d2) {
                maxPair = make_tuple(contour[i], contour[q], d1);
            } else {
                maxPair = make_tuple(contour[i + 1], contour[q], d2);
            }
        }

        return maxPair;
    }

    void sortPts(vector <Point> &(points)) {
        sort(points.begin(), points.end(), [](const Point &p1, const Point &p2) {
            if (p1.x < p2.x) return true;
            else if (p1.x == p2.x) return p1.y < p2.y;
            else return false;
        });
    }

    /**
     * description:压缩点集算法,取三角形的高最长的点,作为分界点. 分治点集为left,right,
     * 递归/stack,直到三角形高小于设定阈值epsilon或者只有两个点
     * @author oswin
     */
    void approxDP::approxDP_(std::vector<cv::Point>& output){

        Range r{0,(int)points.size()-1};
        compress(r);

        swap(result,output);
    }

    #ifndef RECUR
    void approxDP::compress(const Range &range) {
        int first, last;

        auto push_pts = [this](const Range &r) {
            if (!result.empty()) {
                if (result.back().x != points[r.start].x
                    || result.back().y != points[r.start].y) {
                    result.emplace_back(points[r.start]);
                }
            }

            result.emplace_back(points[r.end]);
        };

        std::stack<Range> stack;
        stack.emplace(range);

        while (!stack.empty()) {
            auto top = stack.top();
            stack.pop();

            first = top.start;
            last = top.end;

            if (last - first <= 1) {
                push_pts(top);
                continue;
            }

            int mid = getFarthestPt(top);
            if (mid == -1) {
                push_pts(top);
                continue;
            }

            auto left = Range(first, mid);
            auto right = Range(mid, last);

            stack.emplace(right);
            stack.emplace(left);
        }
    }
    #endif

    #ifdef RECUR
    void approxDP::compress(const Range &range) {

        int first,last;
        first = range.start;
        last = range.end;

        #define PUSH_POINTS                                                                 \
        if (!result.empty())                                                                \
            if (result.back().x != points[first].x || result.back().y != points[first].y)   \
            {                                                                               \
                result.emplace_back(points[first]);                                         \
            }                                                                               \
        result.emplace_back(points[last]);                                                  \

        if (last - first <= 1){
            PUSH_POINTS
            return;
        }

        int mid = getFarthestPt(range);
        if (mid == -1){
            PUSH_POINTS
            return;
        }

        auto left = Range(first,mid);
        auto right = Range(mid,last);

        compress(left);
        compress(right);

        #undef PUSH_POINTS
    }
    #endif

    int approxDP::getFarthestPt(const Range &r) {

        int mid{-1}, h_max{INT_MIN};

        for (int i = r.start + 1; i < r.end - 1; ++i) {
            auto p = points[i];

            auto a = p - points[r.start];
            auto b = points[r.end ] - p;

            auto area = abs(a.cross(b));
            auto h = area / cv::norm(points[r.start] - points[r.end]);

            if (h > h_max){
                h_max = (int)h;
                mid = i;
            }
        }

        if (h_max < epsilon) mid = -1;

        return mid;
    }

    int run(){

        cv::Mat src(100,100,0,Scalar(0));
        cv::circle(src,Point(50,50),30,1,1);

        vector<vector<Point>> cons;
        findContours(src,cons,RETR_EXTERNAL,CHAIN_APPROX_NONE);

        vector<Point> input = cons[0];
        sortPts(input);

        vector<Point> output;
        approxDP ad(cons[0],10);
        ad.approxDP_(output);
        cout<<"output points after being compressed: \n"<<output<<endl;


        /**
         * description:旋转钳算法入口
         * @author ${oswin}
         * @date 03/17
         */
        auto pair = rotatingCalipers(output);
        cout<<"caliper head: "<<get<0>(pair)<<endl;
        cout<<"caliper tail: "<<get<1>(pair)<<endl;
        cout<<"caliper len: "<<get<2>(pair)<<endl;

        visualize(src, pair, cons[0],output);
    }
}
