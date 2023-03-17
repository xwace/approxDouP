#include <utility>

//
// Created by star on 23-3-16.
//

#ifndef DOUGLASPEUCKER_H
#define DOUGLASPEUCKER_H

namespace DSP{

    using maxDisPair = std::tuple<cv::Point, cv::Point, int>;

    int run();

    class approxDP{
    public:
        explicit approxDP(std::vector<cv::Point>points_,double epsilon_)
        :points(std::move(points_)),epsilon(epsilon_){}
        void approxDP_(std::vector<cv::Point>& output);

    private:
        std::vector<cv::Point> points, result;
        double epsilon;

        int getFarthestPt(const cv::Range& range);
        void compress(const cv::Range& range);
    };
}

#endif //DOUGLASPEUCKER_H
