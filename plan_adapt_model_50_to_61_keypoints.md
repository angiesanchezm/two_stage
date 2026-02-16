Plan: Adaptar modelo de 50 a 61 keypoints                                                    
                                                                                              
 Contexto                                                                                     
                                                                                              
 El modelo actual usa 50 keypoints × 3 coordenadas = 150 valores por frame (config trg_size:  
 150). Tus datos tienen 61 keypoints × 3 = 183 valores. Ademas, cada frame en los .skels debe 
  incluir un counter (posicion normalizada 0→1) como ultimo valor, totalizando 184 valores  
 por frame. Tu script de conversion actual escribe 183 valores (sin counter).

 Layout de tus joints:
 - Trunk/body: 0-18 (19 keypoints)
 - Right hand: 19-39 (21 keypoints)
 - Left hand: 40-60 (21 keypoints)

 ---
 Cambios (4 archivos)

 Paso 1: convert_data_sidd_twostage/convert_signjoey_to_twostage.py

 - Agregar columna de counter (frame_idx / (num_frames-1)) como ultimo valor de cada frame
 - Cada frame pasa de 183 → 184 valores
 - Actualizar validacion para verificar multiplos de 184
 - Despues de este cambio: re-correr el script para regenerar los .skels

 Paso 2: Configs/Base.yaml

 - trg_size: 150 → trg_size: 183

 Paso 3: data.py linea 146

 - trg_size=151 (hardcodeado) → trg_size=trg_size (usar variable de linea 67)
 - Esto era BUG #15 del analisis de code quality

 Paso 4: CVT/CVT_training.py lineas 330-340

 - Reshape: (50, 3) → (61, 3)
 - Body: [:8] → [:19] (joints 0-18)
 - Right hand: [29:] → [19:40] (joints 19-39)
 - Left hand: [8:29] → [40:] (joints 40-60)
 - Pesos se mantienen: 0.2 body + 0.4 left + 0.4 right

 ---
 NO necesitan cambios (dimension-agnostic)

 - Conv_model.py — self.mid, //10 slicing, build_model: todo deriva del config
 - batch.py — future prediction usa :-1 para counter, agnostico a dimension
 - helpers.py — DTW usa [:, :-1], sin joints hardcodeados
 - discriminator_Data.py — recibe trg_size del config via build_model
 - dataset.py — opera sobre tensores genericos
 - plot_videos.py — ya esta comentado; se puede actualizar despues si se necesita

 ---
 Verificacion

 1. Verificar que los .skels regenerados tengan valores multiplos de 184 por linea
 2. Correr un epoch de training para verificar que no hay errores de dimension
 3. Checkpoints viejos (trg_size=150) NO seran compatibles — entrenar desde cero

 ---
 Riesgos

 - Checkpoints incompatibles: Los modelos entrenados con 150 no cargan con 183
 - Memoria GPU: Output crece ~21% (1501→1831 con future_prediction=10)


 Los 4 cambios están hechos. Resumen:                          
                                                                                              
  1. convert_signjoey_to_twostage.py — Agrega counter (0→1) como ultimo valor de cada frame
  (183→184 valores/frame), validacion actualizada
  2. Base.yaml — trg_size: 150 → trg_size: 183                                                
  3. data.py — trg_size=151 → trg_size=trg_size (usa variable del config)                     
  4. CVT_training.py — Reshape (50,3) → (61,3), indices: body [:19], right_hand [19:40],      
  left_hand [40:]                                                                             
                    