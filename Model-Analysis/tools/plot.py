
#epoch_1.pth
Evaluating bboxes of pred_instances_3d
mAP: 0.3755                                                                                                                           
mATE: 0.3715
mASE: 0.2824
mAOE: 0.7678
mAVE: 0.7686
mAAE: 0.2659
NDS: 0.4422
Eval time: 95.2s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.731   0.229   0.174   0.347   0.499   0.236 
truck                   0.322   0.421   0.227   0.330   0.899   0.341 
bus                     0.483   0.411   0.201   0.417   1.514   0.478 
trailer                 0.123   0.618   0.247   1.328   0.540   0.227 
construction_vehicle    0.103   0.676   0.436   1.293   0.142   0.335 
pedestrian              0.640   0.331   0.287   1.473   0.921   0.235 
motorcycle              0.272   0.258   0.255   0.688   1.151   0.253 
bicycle                 0.105   0.229   0.274   0.905   0.483   0.024 
traffic_cone            0.506   0.199   0.425   nan     nan     nan   
barrier                 0.470   0.342   0.297   0.131   nan     nan   

#epoch_2.pth
Results writes to work_dirs/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/eval_epoch2/pred_instances_3d/results_nusc.json
Evaluating bboxes of pred_instances_3d
mAP: 0.4440                                                                                                                           
mATE: 0.3312
mASE: 0.2722
mAOE: 0.5268
mAVE: 0.5934
mAAE: 0.2153
NDS: 0.5281
Eval time: 82.9s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.776   0.212   0.171   0.218   0.526   0.219 
truck                   0.399   0.373   0.216   0.244   0.550   0.237 
bus                     0.573   0.411   0.189   0.205   1.297   0.331 
trailer                 0.220   0.596   0.225   1.152   0.420   0.212 
construction_vehicle    0.093   0.677   0.430   1.091   0.136   0.310 
pedestrian              0.787   0.173   0.286   0.532   0.386   0.128 
motorcycle              0.368   0.229   0.252   0.517   1.112   0.268 
bicycle                 0.128   0.194   0.271   0.634   0.319   0.018 
traffic_cone            0.536   0.170   0.386   nan     nan     nan   
barrier                 0.561   0.278   0.296   0.149   nan     nan   