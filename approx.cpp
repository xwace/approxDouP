//
// Created by star on 23-3-17.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <iterator>
#include <deque>
#include "approx.h"

using namespace cv;
using namespace std;
namespace APPROX{

    template<typename T> static int
    approxPolyDP_( const Point_<T>* src_contour, int count0, Point_<T>* dst_contour,
                   bool is_closed0, double eps, AutoBuffer<Range>& _stack )
    {
        #define PUSH_SLICE(slice)       \
        if( top >= stacksz )            \
        {                               \
            _stack.resize(stacksz*3/2); \
            stack = _stack.data();      \
            stacksz = _stack.size();    \
        }                               \
        stack[top++] = slice

        #define READ_PT(pt, pos)        \
        pt = src_contour[pos];          \
        if( ++pos >= count ) pos = 0

        #define READ_DST_PT(pt, pos)    \
        pt = dst_contour[pos];          \
        if( ++pos >= count ) pos = 0

        #define WRITE_PT(pt)            \
        dst_contour[new_count++] = pt

        typedef cv::Point_<T> PT;
        int             init_iters = 3;
        Range           slice(0, 0), right_slice(0, 0);
        PT              start_pt((T)-1000000, (T)-1000000), end_pt(0, 0), pt(0,0);
        int             i = 0, j, pos = 0, wpos, count = count0, new_count=0;
        int             is_closed = is_closed0;
        bool            le_eps = false;
        size_t top = 0, stacksz = _stack.size();
        Range*          stack = _stack.data();

        if( count == 0  )
            return 0;

        eps *= eps;

        if( !is_closed )
        {
            right_slice.start = count;
            end_pt = src_contour[0];
            start_pt = src_contour[count-1];

            if( start_pt.x != end_pt.x || start_pt.y != end_pt.y )
            {
                slice.start = 0;
                slice.end = count - 1;
                PUSH_SLICE(slice);
            }
            else
            {
                is_closed = 1;
                init_iters = 1;
            }
        }

        if( is_closed )
        {
            // 1. Find approximately two farthest points of the contour
            right_slice.start = 0;

            for( i = 0; i < init_iters; i++ )
            {
                double dist, max_dist = 0;
                pos = (pos + right_slice.start) % count;
                READ_PT(start_pt, pos);

                for( j = 1; j < count; j++ )
                {
                    double dx, dy;

                    READ_PT(pt, pos);//获取pt点,同时pos++遍历所有点(除起点)
                    dx = pt.x - start_pt.x;
                    dy = pt.y - start_pt.y;

                    dist = dx * dx + dy * dy;

                    if( dist > max_dist )
                    {
                        max_dist = dist;
                        right_slice.start = j;//找到距离最大的点
                    }
                }

                le_eps = max_dist <= eps;
            }

            // 2. initialize the stack
            if( !le_eps )
            {
                right_slice.end = slice.start = pos % count;
                slice.end = right_slice.start = (right_slice.start + slice.start) % count;

                PUSH_SLICE(right_slice);
                PUSH_SLICE(slice);
            }
            else
                WRITE_PT(start_pt);
        }

        // 3. run recursive process
        while( top > 0 )
        {
            slice = stack[--top];//stack.pop出栈
            end_pt = src_contour[slice.end];
            pos = slice.start;
            READ_PT(start_pt, pos);

            if( pos != slice.end )
            {
                double dx, dy, dist, max_dist = 0;

                dx = end_pt.x - start_pt.x;
                dy = end_pt.y - start_pt.y;

                CV_Assert( dx != 0 || dy != 0 );

                //循环找到最高的点,作为分界点right_slice.start
                while( pos != slice.end )
                {
                    READ_PT(pt, pos);//pos++递增遍历
                    dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

                    if( dist > max_dist )
                    {
                        max_dist = dist;
                        right_slice.start = (pos+count-1)%count;
                    }
                }

                //le_eps是less or equal than: <=
                le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
            }
            else
            {
                le_eps = true;
                // read starting point
                start_pt = src_contour[slice.start];
            }

            if( le_eps )
            {
                WRITE_PT(start_pt);
            }
            else
            {
                //分治前:slice = [left, right]
                //找到分界点right.start开始分两拨: slice = left, right.end赋值
                right_slice.end = slice.end;
                slice.end = right_slice.start;

                PUSH_SLICE(right_slice);
                PUSH_SLICE(slice);
            }
        }

        if( !is_closed )
            WRITE_PT( src_contour[count-1] );

        // last stage: do final clean-up of the approximated contour -
        // remove extra points on the [almost] straight lines.
        is_closed = is_closed0;
        count = new_count;
        pos = is_closed ? count - 1 : 0;
        READ_DST_PT(start_pt, pos);
        wpos = pos;
        READ_DST_PT(pt, pos);

        //这整块循环代码理解为,迭代dst_contour,连续三点的中点太矮的时候删除该点
        //wpos是记录实际保存点的指针;new_count记录dst_contour的长度,删除一个元素,长度-1
        //start_pt前一个点的指针;end_pt后一个点的指针
        for( i = !is_closed; i < count - !is_closed && new_count > 2; i++ )
        {
            double dx, dy, dist, successive_inner_product;
            READ_DST_PT( end_pt, pos );


            dx = end_pt.x - start_pt.x;
            dy = end_pt.y - start_pt.y;
            dist = fabs((pt.x - start_pt.x)*dy - (pt.y - start_pt.y)*dx);
            successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
                                       (pt.y - start_pt.y) * (end_pt.y - pt.y);

            //if代码块相当于[it=dst_contour.erase(it); it++;]; !!!注意下一个点it取在next之后
            if( dist * dist <= 0.5*eps*(dx*dx + dy*dy) && dx != 0 && dy != 0 &&
                successive_inner_product >= 0 )//三点组成的角不为锐角时,中点太矮,删除
            {
                new_count--;//记录dst_contour的长度,删除一个元素,长度-1
                dst_contour[wpos] = start_pt = end_pt;
                if(++wpos >= count) wpos = 0;
                READ_DST_PT(pt, pos);
                //i理解为当前点pt的下标
                i++;//正常情况for(;;i++)保证了当前点每次向后迭代一个,pt移动到next;再加一个i++,则跳到了next后面
                continue;
            }

            //这部分代码相当于(dst_contour::iterator)it++
            dst_contour[wpos] = start_pt = pt;
            if(++wpos >= count) wpos = 0;
            pt = end_pt;
        }

        //用deque方式实现上面的代码
        /*std::deque <Point> dstcontour(new_count);
        copy(dst_contour, dst_contour + new_count, dstcontour.begin());
        dstcontour.emplace_front(*(dst_contour + new_count));
        dstcontour.emplace_back(*dst_contour);

        for (auto it = dstcontour.cbegin() + 1; it != dstcontour.cend() - 1 and it != dstcontour.cend();)
        {
            auto prev = it - 1;
            auto next = it + 1;

            auto dx = next->x - prev->x;
            auto dy = next->y - prev->y;
            auto dist = fabs((it->x - prev->x) * dy - (it->y - prev->y) * dx);
            auto successive_inner_product = (it->x - prev->x) * (next->x - it->x) +
                                            (it->y - prev->y) * (next->x - it->y);

            if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0 && dy != 0 &&
                successive_inner_product >= 0)
                {
                    it = dstcontour.erase(it);
                    it++;//跳过next点
                }

            else it++;
        }

        dstcontour.pop_front();
        dstcontour.pop_back();*/

        if( !is_closed )
            dst_contour[wpos] = pt;

        return new_count;
    }

    void approxPolyDP( InputArray _curve, OutputArray _approxCurve,
                           double epsilon, bool closed )
    {

        if (epsilon < 0.0 || !(epsilon < 1e30))
        {
            CV_Error(CV_StsOutOfRange, "Epsilon not valid.");
        }

        Mat curve = _curve.getMat();
        int npoints = curve.checkVector(2), depth = curve.depth();
        CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F));

        if( npoints == 0 )
        {
            _approxCurve.release();
            return;
        }

        AutoBuffer<Point> _buf(npoints);
        AutoBuffer<Range> _stack(npoints);
        Point* buf = _buf.data();
        int nout = 0;

        if( depth == CV_32S )
            nout = approxPolyDP_(curve.ptr<Point>(), npoints, buf, closed, epsilon, _stack);
        else if( depth == CV_32F )
            nout = approxPolyDP_(curve.ptr<Point2f>(), npoints, (Point2f*)buf, closed, epsilon, _stack);
        else
            CV_Error( CV_StsUnsupportedFormat, "" );

        Mat(nout, 1, CV_MAKETYPE(depth, 2), buf).copyTo(_approxCurve);
    }

    int run(){

        cv::Mat src(100,100,0,Scalar(0));
        cv::circle(src,Point(50,50),30,1,1);

        std::vector<std::vector<Point>> cons;
        findContours(src,cons,RETR_EXTERNAL,CHAIN_APPROX_NONE);

        std::vector<Point> input = cons[0], output;
        double epsilon = 5;

        APPROX::approxPolyDP(input, output, epsilon, true);

        std::cout<<"output points after being compressed: \n"<<output<<std::endl;

        {
            Mat out = src.clone().setTo(0);
            cvtColor(out,out,COLOR_GRAY2BGR);
            for (const auto& c: cons[0]) {
                out.at<Vec3b>(c) = Vec3b(255,255,255);
            }

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
    }
}
