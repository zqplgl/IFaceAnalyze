#ifndef IVEHICLEANALYZE_H_
#define IVEHICLEANALYZE_H_


#include <Itracker.h>
class IFaceAnalyze
{
    public:
        virtual void input(const string &filepath,const string &pic_save_dir,const int save_frame_skip)=0;
        virtual void process()=0;
        virtual ~IFaceAnalyze(){}
};

IFaceAnalyze *CreateIFaceAnalyze(const string &model_dir,const int gpu_id,const int frameskip, const int numnull);

#endif

