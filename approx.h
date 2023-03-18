//
// Created by star on 23-3-17.
//

#ifndef APPROX_H
#define APPROX_H

namespace APPROX{
    void approxPolyDP( cv::InputArray _curve, cv::OutputArray _approxCurve,double epsilon, bool closed );
    int run();
}

#endif //APPROX_H
