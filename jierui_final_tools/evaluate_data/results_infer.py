import numpy as np
import os
import glob
import motmetrics as mm
import sys
from utils.evaluation import Evaluator


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def eval_mota(data_root, txt_path):
    accs = []
    # seqs = sorted([s for s in os.listdir(data_root) if s.endswith('FRCNN')])
    seqs = sorted([s for s in os.listdir(data_root)])
    # seqs = sorted([s for s in os.listdir(data_root) if s.endswith('SDP')])
    # seqs = sorted([s for s in os.listdir(data_root) if s.endswith('DPM')])
    # seqs = sorted([s for s in os.listdir(data_root)])
    for seq in seqs:
        video_out_path = os.path.join(txt_path, seq + '.txt')
        evaluator = Evaluator(data_root, seq, 'mot')
        accs.append(evaluator.eval_file(video_out_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    # np.savetxt('temp_np.txt',strsummary,delimiter='\t')
    print(strsummary)
    with open('temp.txt','w') as f:
        print(strsummary,file=f,sep='\t',end='\n')
    return summary


def evaluate(data_root,save_folder):
    summary = eval_mota(data_root, save_folder)
    return {'MOTA_OVERALL':summary.mota['OVERALL']}
if __name__ == '__main__':
    data_root =r'/Users/lisushang/Downloads/jierui2024_MOTformat_RGB/train/'
    save_folder =r"/Users/lisushang/PycharmProjects/tracking_tools_mac/jierui_final_tools/temp/track_outputs"
    evaluate(data_root, save_folder)
    print(save_folder)