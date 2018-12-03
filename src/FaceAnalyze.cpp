#include <opencv2/opencv.hpp>
#include <math.h>

#include <IFaceAnalyze.h>
#include <Imtcnn.h>
#include <FaceAnalyze.h>

using namespace cv;

FaceAnalyze::FaceAnalyze(const string& model_dir,const int gpu_id, const int frameskip,const int numnull):gpu_id(gpu_id),frameskip(frameskip),numnull(numnull)
{
    detector = CreateObjZoneMTcnnDetector(model_dir);
    confidence_threshold.push_back(0.6);
    confidence_threshold.push_back(0.6);
    confidence_threshold.push_back(0.7);
    tracker = tracker::CreateITrackers();
}

void FaceAnalyze::input(const string &filepath,const string &pic_save_dir,const int save_frame_skip)
{
    capture.open(filepath);
    string temp_dir = "";
    if (pic_save_dir[pic_save_dir.size()-1]=='/')
        temp_dir = pic_save_dir;
    else
        temp_dir = pic_save_dir + "/";

    big_picture_dir = temp_dir + "bigpicture";
    face_picture_dir = temp_dir + "facepicture";
    this->save_frame_skip = save_frame_skip;
}

void FaceAnalyze::transform(std::vector<BoundingBox>& faceinfos, vector<tracker::Object> &tracker_objs)
{
    tracker_objs.clear();
    for(int i=0; i<faceinfos.size(); ++i)
    {
        tracker::Object obj;
        obj.first.x = faceinfos[i].x1;
        obj.first.y = faceinfos[i].y1;
        obj.first.width = faceinfos[i].x2 - faceinfos[i].x1;
        obj.first.height = faceinfos[i].y2 - faceinfos[i].y1;
        obj.second = 1;
        tracker_objs.push_back(obj);
    }
}

void addRectangle(cv::Mat &img,const vector<BoundingBox> &res)
{
	for (int k = 0; k < res.size(); k++)
	{
        cv::rectangle(img, cv::Point(res[k].x1, res[k].y1), cv::Point(res[k].x2, res[k].y2), cv::Scalar(0, 255, 255), 2);
        for(int i = 0; i < 5; i ++)
            cv::circle(img, cv::Point(res[k].points_x[i], res[k].points_y[i]), 2, cv::Scalar(0, 255, 255), 2);
	}
}

BoundingBox getFaceInfo(const vector<BoundingBox> &faceinfos,const cv::Rect &rect)
{
    for(int i=0; i<faceinfos.size(); ++i)
    {
        if(faceinfos[i].x1==rect.x && faceinfos[i].y1==rect.y)
        {
            return faceinfos[i];
        }
    }
}

Mat align(const Mat &im, const vector<int> &points)
{
    float center_x = (points[0]+points[1])/2.0f;
    float center_y = (points[2]+points[3])/2.0f;
    float dx = points[1] - points[0];
    float dy = points[3] - points[2];

#define PI 3.14159265
    float angle = atan2(dy,dx)*180/PI;
    Mat matrix = getRotationMatrix2D(Point(center_x,center_y),angle,1);
    Mat im_align;
    warpAffine(im,im_align,matrix,im.size());

    return im_align;
}

Mat FaceAnalyze::get_align_face(const Mat &im, const BoundingBox &faceinfo,int &flag)
{
    float pad_scale = 0.3;
    int w = faceinfo.x2 - faceinfo.x1;
    int h = faceinfo.y2 - faceinfo.y1;
    int pad_w = w * pad_scale;
    int pad_h = h * pad_scale;

    int x1 = faceinfo.x1 - pad_w;
    int y1 = faceinfo.y1 - pad_h;
    int x2 = faceinfo.x2 + pad_w;
    int y2 = faceinfo.y2 + pad_h;

    if(x1<0) x1 = 0;
    if(y1<0) y1 = 0;
    if(x2>=im.cols) x2 = im.cols -1;
    if(y2>=im.rows) y2 = im.rows -1;

    Mat face = im(Rect(x1,y1,x2-x1,y2-y1)).clone();

    vector<int> points;
    points.push_back(faceinfo.points_x[0] - x1);
    points.push_back(faceinfo.points_x[1] - x1);
    points.push_back(faceinfo.points_y[0] - y1);
    points.push_back(faceinfo.points_y[1] - y1);
    Mat face_align = align(face,points);

    vector<BoundingBox> faceinfos;
    detector->detection(face_align,confidence_threshold,faceinfos);
    if(faceinfos.size()==0)
    {
        flag==1;
        return face;
    }

    x1 = faceinfos[0].x1;
    y1 = faceinfos[0].y1;
    x2 = faceinfos[0].x2;
    y2 = faceinfos[0].x2;

    if(x1<0) x1 = 0;
    if(y1<0) y1 = 0;
    if(x2>=face_align.cols) x2 = face_align.cols -1;
    if(y2>=face_align.rows) y2 = face_align.rows -1;

    return face_align(Rect(x1,y1,x2-x1,y2-y1)).clone();
}


void FaceAnalyze::process()
{
    Mat frame;
    int null_num = 0;
    int skip_num = 0;
    vector<BoundingBox> faceinfos;
    vector<tracker::Object> tracker_objs;

    int frame_index = 0;

    vector<tracker::Tracker> trackers_over;
    vector<tracker::Tracker> trackers_running;

    char pic_save_path[1024];
    while(capture.read(frame))
    {
        trackers_over.clear();
        frame_index++;
        detector->detection(frame,confidence_threshold,faceinfos);

        transform(faceinfos,tracker_objs);
        tracker->Update(tracker_objs,frame_index);
        tracker->getTracks(trackers_over,trackers_running);

        if(frame_index%save_frame_skip==0 && trackers_running.size())
        {

            int flag = 1;

            vector<BoundingBox> infos;
            for(int i=0; i<trackers_running.size(); ++i)
            {
                if(trackers_running[i].end_frame-1!=frame_index)
                    continue;

                if(flag)
                {
                    sprintf(pic_save_path,"%s/%d.jpg",big_picture_dir.data(),frame_index);
                    imwrite(string(pic_save_path),frame);
                    flag = 0;
                }

                tracker::Track &rects = trackers_running[i].track;
                cv::Rect rect = rects[rects.size()-1];
                BoundingBox faceinfo = getFaceInfo(faceinfos,rect);

                int flag1 = 0;
                cout<<"face_align before************************"<<frame_index<<endl;
                Mat face_align = get_align_face(frame,faceinfo,flag1);
                cout<<"face_align after************************"<<frame_index<<endl;
                if(flag1)
                    continue;

                int x1 = faceinfo.x1;
                int y1 = faceinfo.y1;
                int x2 = faceinfo.x2;
                int y2 = faceinfo.y2;
                int id = trackers_running[i].id;
                int size = (y2-y1)*(x2-x1);
                
                sprintf(pic_save_path,"%s/%d_%d_%d-%d-%d-%d_%d.jpg",face_picture_dir.data(),frame_index,id,x1,y1,x2,y2,size);

                imwrite(string(pic_save_path),face_align);
            }
        }
        cout<<"process frame "<<frame_index<<" successful"<<endl;
    }

}

IFaceAnalyze *CreateIFaceAnalyze(const string &model_dir,const int gpu_id, const int frameskip, const int numnull)
{
    IFaceAnalyze *ptr=nullptr;
    ptr = new FaceAnalyze(model_dir,gpu_id,frameskip,numnull);

    return ptr;
}
