| time | model                                 | hypeparameters                                                                                                                 | coremetric | loss     | lambada_openai | parameters | flops | commit                                   |   |
|------|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|------------|----------|----------------|------------|-------|------------------------------------------|---|
|      | nanochat                              |                                                                                                                                | 0.2457     |          |                |            |       |                                          |   |
| 0624 | d14,hiddensize 876, untiled embedding | lr 3e-3 embedding lr 3e-3 muon_lr 0.02 adaw  betas=(0.9, 0.95), wd ~0.13 step 4688 warmup 0.05 target-param-data-ratio12       |            | 3.13     | 0.2137         | 4687       |       | 26c131d4e0e2832025438ac9a5bfa7033814b909 |   |
| 0625 | d28 hiddensize 1024 untiled embedding | lr 3e-3,  wd0.02, muon_lr 3e-3, embeddinglr  0.03   betas=(0.9, 0.95)  warmup 0.1  target-param-data-ratio12   steps 10273     | 0.1413     | 2.925588 | 0.2808         |            |       | 4e5ff4213ad3a65558b383c99ab130bf79daf51e |   |
| 0625 | d28 hiddensize 1024 untiled embedding | lr 3e-3,  wd0.02, muon_lr  3e-3, **embeddinglr  0.3**   betas=(0.9, 0.95)  warmup 0.1  target-param-data-ratio12   steps 10273 | 0.1269     | 2.932613 | 0.2729         |            |       | 4e5ff4213ad3a65558b383c99ab130bf79daf51e |   |
| 0625 | d28 hiddensize 1024 untiled embedding | lr 3e-3,  wd0.02, muon_lr 3e-3, **embeddinglr  0.03**   betas=(0.9, 0.95)  warmup 0.1  target-param-data-ratio12   steps 10273 | 0.1445     | 2.932555 | 0.2779         |            |       | 4e5ff4213ad3a65558b383c99ab130bf79daf51e |   |
| 0625 | d14 hiddensize 1024 untiled embedding | lr 3e-3,  wd0.02, muon_lr 3e-3, embeddinglr  0.3   betas=(0.9, 0.95)  warmup 0.1  target-param-data-ratio12   steps 05904      | 0.1202     | 3.087198 | 0.2395         |            |       |                                          |   |


run_datetime_bj,git_commit,depth,lr,weight_decay,warmup_ratio,muon_lr,hidden_size,embedding_lr,target_param_data_ratio,grad_max_norm,params_total,num_iterations,tokens_trained,final_loss,val_bpb,core_score,train_time_sec
2026-06-25 16:24:51,abca99f939da1866e701f138d1cb7486e0d4205c,14,0.003,0.02,0.1,0.02,1024,0.3,12,-1.0,0,5904,3095396352,3.204387,,0.0966,15219
2026-06-25 20:38:30,abca99f939da1866e701f138d1cb7486e0d4205c,14,0.003,0.02,0.1,0.02,1024,0.3,12,1.0,0,5904,3095396352,3.230884,,0.0995,15225
2026-06-26 00:52:16,abca99f939da1866e701f138d1cb7486e0d4205c,14,0.003,0.02,0.1,0.02,1024,0.03,12,-1.0,0,5904,3095396352,3.221025,,0.087,15215
2026-06-26 05:05:51,abca99f939da1866e701f138d1cb7486e0d4205c,14,0.003,0.02,0.1,0.02,1024,0.03,12,1.0,0,5904,3095396352,3.250235,,0.0886,15258


run_datetime_bj,git_commit,depth,lr,weight_decay,warmup_ratio,muon_lr,hidden_size,embedding_lr,target_param_data_ratio,grad_max_norm,params_total,num_iterations,tokens_trained,final_loss,val_bpb,core_score,train_time_sec
2026-06-25 18:06:59,abca99f939da1866e701f138d1cb7486e0d4205c,28,0.003,0.02,0.1,0.02,1024,0.3,12,1.0,0,10273,5386010624,3.123444,,0.113,16126
2026-06-25 22:35:45,abca99f939da1866e701f138d1cb7486e0d4205c,28,0.003,0.02,0.1,0.02,1024,0.03,12,1.0,0,10273,5386010624,3.08631,,0.1235,16170
2026-06-26 03:05:16,abca99f939da1866e701f138d1cb7486e0d4205c,28,0.003,0.02,0.1,0.002,1024,0.3,12,1.0,0,10273,5386010624,2.91826,,0.1385,16051
