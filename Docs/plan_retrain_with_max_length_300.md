Plan: Retrain with max_length=300 (all sequences up to 300 frames)                         
                                                                                            
 Context                                                                                  
                                                                                          
 Currently the model filters out sequences >100 frames, discarding ~60% of training data.   
 The user wants to retrain with max_length=300, which covers >97% of sequences.             
                                                                                            
 Key relationship: trg_length = max_length + 2 (because collate pads +12 and Batch strips   
 -10 for future_prediction)                                                                 
                                                                                            
 Changes                                                                                    
                                                                                            
 1. data.py:274 — Update filter threshold                                                   
                                                                                            
 # Current:  if len(trg_frames) > 100:
 # New:      if len(trg_frames) > 300:

 2. data_operate/dataset.py:20 — Update collate_fn max_length

 # Current:  max_length = 100
 # New:      max_length = 300
 Also remove dead comment on line 21 (# max_length = 112).

 3. Configs/Base.yaml:89 — Update trg_length

 # Current:  trg_length: 102
 # New:      trg_length: 302
 Also reduce batch_size from 128 to 32 (sequences ~3x longer = ~3x more memory).

 4. helpers.py:244 — Update DTW unpadding

 # Current:  max_length = 100
 # New:      max_length = 300

 5. CVT/CVT_training.py:545 — Update video rendering unpadding

 # Current:  max_length = 100
 # New:      max_length = 300

 6. CVT/Conv_model.py:243 — Update default fallback (cosmetic)

 # Current:  cfg.get("trg_length", 102)
 # New:      cfg.get("trg_length", 302)

 Files modified (6 files)

 - data.py (line 274)
 - data_operate/dataset.py (line 20-21)
 - Configs/Base.yaml (lines 35, 89)
 - helpers.py (line 244)
 - CVT/CVT_training.py (line 545)
 - CVT/Conv_model.py (line 243)

 Important notes

 - Checkpoint incompatible: temporal_conv changes from Conv1d(18,102,...) to
 Conv1d(18,302,...). Must train from scratch.
 - batch_size: Reduce from 128 to 32 due to ~3x memory increase per sample.
 - Sequences >300 frames (train has some up to 475) will still be filtered out (~3% of
 train data).

 Verification

 1. Run training: python __main__.py CVT ./Configs/Base.yaml
 2. Confirm no crashes on first epoch
 3. Check that more sequences are loaded (should see ~6,900 train vs previous 2,792)