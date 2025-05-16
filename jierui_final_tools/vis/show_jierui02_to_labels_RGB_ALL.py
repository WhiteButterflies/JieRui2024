#2019-04-29 Xingyu Zeng
#
import os
import cv2
import numpy as np
import json

def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False



def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).
    Args:
        detections
    Returns:
        list: list containing the detections for each frame.
    """

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)

    end_frame = int(np.max(raw[:, 0]))
    for i in range(1, end_frame + 1):
        idx = raw[:, 0] == i

        person_id=raw[idx,1]
        # print(i,person_id)
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = raw[idx, 6]
        class_name=raw[idx,7]
        dets = []
        for bb, s,p_id ,c in zip(bbox, scores,person_id,class_name):
            # 源代码没有除以2  由于下载的视频相较与标签分辨率缩小了一半，所以除以2
            # dets.append({'bbox': (int(bb[0] / 2), int(bb[1] / 2), int(bb[2] / 2), int(bb[3] / 2)), 'score': s})
            dets.append({'bbox': (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])), 'score': s,'p_id':int(p_id),'class_id':c})
        data.append(dets)

    return data


# COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),
COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),( 152,251,152),(  0,128,  0),
            (210,105, 30),(220, 20, 60),(192,192,192),(255,228,196),( 50,205, 50),
            (139,  0,139),(100,149,237),(138, 43,226),(238,130,238),(255,  0,255),
            (  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),
            # (255,239,213),(199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),
            # (255,239,213),(199, 21,133),(124,252,  0),(147,112,219),(0,255,0),
            (255,239,213),(0,0,255),(124,252,  0),(147,112,219),(0,255,0),
            (176,196,222),( 65,105,225),(173,255, 47),(255, 20,147),(219,112,147),
            # (0,255,0),( 65,105,225),(173,255, 47),(255, 20,147),(219,112,147),
            (186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),
            (255,255,224),(128,128,128),(105,105,105),( 64,224,208),(205,133, 63),
            (  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),(250,240,230),
            (152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),
            (  0,250,154),(245,255,250),(240,230,140),(245,222,179),(  0,139,139),
            (143,188,143),(255,  0,  0),(240,128,128),(102,205,170),( 60,179,113),
            ( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def genColorByPid(pid):
    return COLORS_10[pid % len(COLORS_10)]

def add_mask_img(img0,mask):
    a=0.95
    b=0.3
    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            if sum(mask[i][j])==0:
                mask[i][j]=img0[i][j]*a+mask[i][j]*(1-a)
            else:
                mask[i][j] = img0[i][j] * b + mask[i][j] * (1 - b)
    return mask
def main():


    show_time=1000
    show_time=1
    show_time=0
    obj_count=0
    objs=['MOT17-02-DPM']
    objs=['MOT17-02-FRCNN']
    objs=['/data1/lqh/jierui2024_train/0012/visible']
    objs=['0012']
    objs=['0020']
    objs=['0061']
    # test_rate=0.2
    # test_rate=0.8

    # objs=['MOT17-04-FRCNN']
    # objs=['MOT17-05-FRCNN']
    # objs=['MOT17-10-FRCNN']
    # objs=['MOT17-11-FRCNN']
    # objs=['MOT17-13-FRCNN']
    # objs=['MOT17-13-FRCNN']
    # objs=['MOT17-09-FRCNN']
    # objs=['MOT17-09-FRCNN']
    all_targets=[]
    all_languates=[]
    # json_path = r'/home/zzb/zzb/pythonfile/mot/2024/aaai/baseline/LaMOT/annotations_v1/train/MOT17'
    # all_json_objs=os.listdir(json_path)
    # all_json_objs.sort()
    for obj in objs:
        obj_count+=1
        print('{}/{}'.format(obj_count,len(objs)))
        # obj='MOT17-05-FRCNN'
        video_path = r'/data2/zzb/data/MOT/MOT17/train/{}'.format(obj)
        video_path = r'/data1/lqh/jierui2024_train/{}/visible/'.format(obj)
        video_path = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/{}/'.format(obj)
        video_path_gt = r'/data2/zzb/data/MOT/MOT17/train/{}/gt/gt.txt'.format(obj)
        video_path_gt = r'/data1/lqh/jierui2024_train/{}/visible/gt/gt.txt'.format(obj)
        video_path_gt = r'/data1/lqh/jierui24_final_GT/train/{}/gt/gt_mask.txt'.format(obj)
        video_path_gt = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/{}/gt/gt.txt'.format(obj)
        video_path_gt = r'/Users/lisushang/Downloads/jierui24_final_RGB/train/{}/det/det.txt'.format(obj)
        # video_path_gt = r'/data2/zzb/model/MOT/2024/aaai/LG-MOT/outputs/exp_test/experiments/example_mot17_training_fastreid_msmt_BOT_R50_ibn/oracle/mot_files/{}.txt'.format(obj)

        # print(temp_json['language'])
        # print(temp_json['targets'])
        video_path_img1='{}/{}'.format(video_path,'image')
        # video_path_img1='{}/{}'.format(video_path,obj)
        img1_path=os.listdir(video_path_img1)
        img1_path.sort()
        # dets=load_mot(video_path_det)
        # print(video_path_gt)
        dets=load_mot(video_path_gt)

        show_name = str(obj)
        # targets=temp_json['targets']
        # language=temp_json['language']
        cv2.namedWindow(show_name,cv2.WINDOW_NORMAL)
        for frame_num, detections_frame in enumerate(dets, start=1):
            #读取视频
            # return_value,frame=vid.read()
            frame=cv2.imread('{}/{}'.format(video_path_img1,img1_path[frame_num-1]))

            for a in range(len(detections_frame)):
                bbox=detections_frame[a]["bbox"]
                p_id=detections_frame[a]["p_id"]
                class_id=int(detections_frame[a]["class_id"])



                # rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None):
                #将解析处理的矩形框，绘制在视频上，实时显示
                # cv2.rectangle(frame,bbox[:2],bbox[2:],(255,0,0),2)
                # out_lauguages=""
                # for i1 in range(len(frames_targets)):
                #     if p_id in frames_targets[i1]:
                #         out_lauguages+=frames_languages[i1]
                # # print(out_lauguages)
                #     # temp_lauguages=frames_languages[i1]

                cv2.rectangle(frame,bbox[:2],bbox[2:],genColorByPid(p_id),3)
                # cv2.rectangle(frame_back,bbox[:2],bbox[2:],genColorByPid(p_id),-1)
                # frame=cv2.addWeighted(frame,1,frame_back,0.9,1)
                # frame=add_mask_img(frame,frame_back)
                # print('222222222222222222')
                # exit()
                show_font_num=2

                cv2.putText(frame,'{},{}'.format(p_id,class_id),(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,2,genColorByPid(p_id),2)
                # cv2.putText(frame,'{}'.format(out_lauguages),(bbox[0],bbox[1]-50),cv2.FONT_HERSHEY_COMPLEX,2,genColorByPid(p_id),2)

            cv2.putText(frame, '#{}/{}'.format(frame_num, len(img1_path)), (20, 50), cv2.FONT_HERSHEY_COMPLEX,
                        show_font_num, (0, 255, 255), 3)

            cv2.resizeWindow(show_name,int(frame.shape[1]/2),int(frame.shape[0]/2))
            cv2.imshow(show_name, frame)
            cv2.moveWindow(show_name, 20, 20)
            # cv2.imwrite(r'D:\pythonfile\NFS_benchmark\test\2022\0517\temp\{:04d}.jpg'.format(frame_num),frame)
            # save_path=r'/home/zzb/zzb/pythonfile/mot/deep/results/mot/mot17_02_2'
            # mkdir(save_path)

            # print(frame_num)
            # cv2.imwrite(r'{}/{:04d}.jpg'.format(save_path,frame_num),frame)
            # 键盘控制视频播放  waitKey(x)控制视频显示速度
            ch = cv2.waitKey(show_time)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            elif ch == ord(' '):
                cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()