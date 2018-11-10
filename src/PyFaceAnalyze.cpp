//
// Created by zqp on 18-8-1.
//

#include<boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <FaceAnalyze.h>
#include <opencv2/opencv.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(_FaceAnalyze)
{
    class_<FaceAnalyze>("FaceAnalyze",init<const string&,const int,const int,const int>())
            .def("input",&FaceAnalyze::input)
            .def("process",&FaceAnalyze::process);

}
