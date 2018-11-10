#ifndef PROJECT_FACEANALYZE_H
#define PROJECT_FACEANALYZE_H

#include <opencv2/opencv.hpp>

#include <IFaceAnalyze.h>
#include <Imtcnn.h>

using namespace cv;

class FaceAnalyze : public IFaceAnalyze 
{
public:
    FaceAnalyze(const string &model_dir, const int gpu_id, const int frameskip, const int numnull);

    virtual void input(const string &filepath,const string &pic_save_dir,const int save_frame_skip);
    virtual void process();

    Mat get_align_face(const Mat &im, const BoundingBox &faceinfo,int &flag);


private:
    void transform(std::vector<BoundingBox>& faceinfos, vector<tracker::Object> &tracker_objs);

private:
    IObjZoneDetect *detector = nullptr;
    tracker::ITrackers *tracker = nullptr;
    int frameskip = 0;
    int numnull = 0;
    int gpu_id = 0;
    VideoCapture capture;
    int capindex = 0;
    vector<float> confidence_threshold;
    string big_picture_dir = "";
    string face_picture_dir = "";
    int save_frame_skip = 0;
};


#endif
