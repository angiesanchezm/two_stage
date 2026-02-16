# Code Quality Issues - Analisis Interno del Codigo

**Repo:** SG_Latent
**Total issues encontradas:** 63 (20 BUG, 19 WARNING, 24 CLEANUP)

---

## BUGS (20)

### 1. CVT_prediction.py return count mismatch — Crash en test time

`CVT_test` en `CVT_training.py:753-754` intenta desempaquetar 11 valores del return de `pre_validate_on_data` (`CVT_prediction.py:127-128`), pero solo retorna 9. Crashea con `ValueError: not enough values to unpack`.

### 2. batch.py:29 — `future_prediction = 10` hardcodeado

El parametro `model` se pasa a `Batch()` pero nunca se lee su `.future_prediction`. Si cambias el config a otro valor, Batch seguiria usando 10.

### 3. batch.py:27 — `just_count_in = False` hardcodeado

Mismo problema: ignora el config. Si `just_count_in` fuera `True` en el YAML, Batch seguiria usando `False`.

### 4. model.py:132-135 — `decode()` pasa longitudes incorrectas al BiLSTM

```python
len = []
for i in range(encoder_output.size(0)):
    len.append(encoder_output.size(0))
```

Crea una lista donde cada elemento es `batch_size` en vez de `sequence_length`. Con batch_size=16 y seq_len=102, `lgt` seria `[16, 16, ..., 16]` (16 veces) en vez de `[102, 102, ..., 102]`.

### 5. CVT_training.py:151-159 — Checkpoint continuation logic invertida

Cuando `continue=True` y encuentra un checkpoint automaticamente (`ckpt is None`), linea 155 obtiene el checkpoint pero **no llama** a `init_from_checkpoint`. Solo logea "Can't find checkpoint". Cuando `ckpt is not None` (rama else), si carga. La logica esta invertida.

### 6. CVT_training.py:808 — Llama a `train()` que no existe

El bloque `__main__` al final del archivo llama `train(cfg_file=args.config)` pero la funcion se llama `CVT_train`. Crashea con `NameError` si se ejecuta directamente.

### 7. Conv_model.py:150 — `//10` hardcodeado en `AEdecode`

```python
skel_out = torch.cat((skel_out[:, :, :skel_out.shape[2] //10], skel_out[:, :, -1:]), dim=2)
```

Asume `future_prediction=10`. Si cambia en el config, produce dimensiones incorrectas.

### 8. plot_videos.py:357 — `alter_DTW_timing` retorna variable equivocada

Calcula la alineacion DTW en `new_pred_seq` (linea 333) pero retorna `pred_seq` (el original, linea 357). La funcion es un no-op: el alineamiento DTW se calcula y se descarta.

### 9. data.py:286 — `trg_line != ''` siempre es True

Despues de `trg_line = trg_line.split(" ")` (linea 262), `trg_line` es una lista. Comparar lista con string (`!= ''`) siempre da `True` en Python. El filtro no hace nada.

### 10. CVT_training.py:573-576 — `is not "</s>"` usa identidad, no igualdad

```python
if input[1] is not "</s>":
```

Deberia ser `!= "</s>"`. `is not` compara identidad de objeto, no valor. Puede funcionar por string interning en CPython, pero no es garantizado.

### 11. CVT_training.py:292-296 — `self.pre_model` nunca definido

Cuando `gaussian_noise=True`, accede a `self.pre_model.out_stds`. Pero `self.pre_model` nunca se asigna en `__init__` ni en ningun otro metodo. Crashea con `AttributeError`.

### 12. CVT_training.py:602,631 — `self.latent_loss` y `self.latent_optimizer` nunca definidos

`_train_batch` usa ambas variables que no existen. Crashea si se llama.

### 13. CVT_training.py:359-361 — Gradient clipping antes de backward

```python
self.clip_grad_fun(params=self.model.parameters())  # <-- antes
self.optimizer.zero_grad()
batch_loss.backward()
self.optimizer.step()
```

Clipea gradientes de la iteracion anterior (stale). Deberia ser despues de `backward()` y antes de `step()`.

### 14. Conv_model.py:143 y model.py:132 — `len = []` sombrea built-in

`len` es un built-in de Python. Asignar `len = []` lo sombrea. Si cualquier codigo posterior en el mismo scope necesitara `len()`, crashearia con `TypeError: 'list' object is not callable`.

### 15. data.py:146 — test_data hardcodea `trg_size=151`

Deberia usar `cfg["model"]["trg_size"]+1` como train y dev. Si `trg_size` cambiara en el config, test_data quedaria desincronizado.

### 16. helpers.py:299 — `dtw_distance` sombrea import `dtw`

`from dtw import dtw` importa la funcion. Luego `dtw_distance` crea `dtw = torch.zeros(...)` que sombrea el import. Si intentara llamar a la funcion `dtw` de la libreria dentro de esa funcion, fallaria.

### 17. search.py:89,92 — `just_count_in` causa doble-append a `ys`

Cuando `model.just_count_in` es True, linea 89 appends a `ys`, y luego linea 92 tambien (sin `else`). Se anaden dos frames por iteracion en vez de uno.

### 18. initialization.py:212-213 — Accede a `model.encoder.rnn` que no existe

`isinstance(model.encoder.rnn, nn.LSTM)` — pero el encoder es `TransformerEncoder` que no tiene `.rnn`. No crashea en practica porque la rama nunca se alcanza, pero es un bug latente.

### 19. builders.py:166-203 — `NoamScheduler` sin `load_state_dict`

`state_dict()` retorna `None`. No existe `load_state_dict()`. Si el scheduler fuera NoamScheduler, `CVT_training.py:253` crashearia al intentar cargar el state dict. No se activa porque el config usa `"plateau"`.

### 20. CVT_prediction.py:74-75 — `tsne()` llamada con argumentos incorrectos

Comentada pero indicativa: `tsne(midd)` pasa 1 argumento, pero `tsn.py:tsne()` requiere 2 (`input_data, name`). Crashearia si se descomentara.

---

## WARNINGS (19)

### 21. loss.py:106-114 — `HuberLoss` ignora config

Todas las ramas (`l1`, `mse`, else) crean `nn.HuberLoss()`. La seleccion de loss por config no tiene efecto.

### 22. discriminator_Data.py:43 — `32` hardcodeado en Linear

`nn.Linear(pose_time_dim // 4 * 32, embedding_dim)` — el `32` depende del output de conv layers. Si `embedding_dim` cambia, el `32` necesitaria recalcularse.

### 23. data_operate/dataset.py:20 — `max_length = 100` hardcodeado

Deberia derivarse del config (`trg_length`).

### 24. data_operate/dataset.py:22 — `left_pad = 6` hardcodeado

Numero magico que aparece en multiples archivos.

### 25. CVT_training.py:402,404 — TensorBoard escribe misma key dos veces

`add_scalar("train/train_batch_loss", ...)` duplicado. La segunda escritura sobreescribe la primera.

### 26. CVT_training.py:329-332 — `.cuda()` sin verificar disponibilidad

```python
torch.from_numpy(...).cuda()
```

Deberia respetar `self.use_cuda`.

### 27. Base.yaml:79 vs model.py:72 — Default de `gaussian_noise` contradictorio

Config dice `False`. Pero `model.py:72` tiene `model_cfg.get("gaussian_noise", True)` — default `True`. Si faltara la key, el comportamiento seria opuesto al esperado.

### 28. CVT_prediction.py:118-119 — Validacion limitada a ~20 muestras

```python
if batches == math.ceil(20 / batch_size): break
```

Solo valida con ~20 muestras, no el dataset completo.

### 29. batch.py:24,69 — `use_cuda = True` hardcodeado pero no usado

Define `use_cuda = True` en linea 24 pero usa `torch.cuda.is_available()` en linea 69. Variable muerta.

### 30. model.py:155,192 — Accede a atributos de una lista

`batch.trg_input` y `batch.trg` usan notacion de atributos, pero `Batch()` retorna una lista. Crashearia si `get_loss_for_batch` o `run_batch` se llamaran.

### 31-32. Numeros magicos dispersos

- `6` (left_pad) en: `CVT_training.py:543`, `helpers.py:245`, `dataset.py:22`
- `100` (max_length) en: `CVT_training.py:543`, `helpers.py:244`, `dataset.py:20`, `data.py:282`

Deberian ser constantes compartidas o valores del config.

### 33. data.py:71 — `EOS_TOKEN` redefinido localmente

Importa `EOS_TOKEN` de `constants` y luego lo redefine como `EOS_TOKEN = '</s>'` en linea 71. Mismo valor pero sombrea el import.

### 34. Dos clases `Model` con interfaces incompatibles

- `CVT/Conv_model.py:Model.forward(src)` — toma source tokens
- `model.py:Model.forward(trg_input, trg_mask)` — toma target input

Importar `Model` de `model.py` en `CVT_training.py:24` mientras se usa el de `Conv_model.py` genera confusion.

### 35. vocab_dataset.py:22 — Indices de vocabulario inestables

`word2idx` se construye desde `Counter.most_common()`, ordenado por frecuencia. Los indices pueden cambiar si los datos cambian. No hay serializacion para garantizar consistencia entre ejecuciones.

### 36. data.py:252-257 — Side effect: escribe a `test.skels` durante data loading

Abre `test.skels` en modo append y escribe datos de skeleton durante cada carga de datos, incluso durante training. El archivo crece indefinidamente.

### 37. data_operate/vocab_dataset.py type hints — Dict vs Vocabulary

`build_vocab` retorna un `dict`, pero los type hints en `Conv_model.py` dicen `Vocabulary`. `len(src_vocab)` funciona en dict pero es misleading.

### 38. Conv_model.py:224 — `trg_linear` definido pero nunca usado en build_model

Se crea pero no se pasa al constructor de Model. Solo `latent_embed` se pasa.

### 39. discriminator_Data.py:49 — `self.softmax` definido pero nunca usado

`torch.nn.LogSoftmax(dim=-1)` creado pero nunca llamado en `forward()`.

---

## CLEANUP (24)

### Dead imports por archivo

| Archivo | Imports no usados |
|---------|-------------------|
| `CVT_training.py:3,11,15,24` | `pickle`, `Tensor`, `Dataset`, `Model` (de model.py) |
| `CVT_training.py:25` | `RegLoss`, `HuberLoss` (importados pero se usan `torch.nn.MSELoss`/`HuberLoss`) |
| `Conv_model.py:17` | `Vocabulary` (type hint misleading) |
| `encoders.py:3,6,7,13` | `torch`, `pack_padded_sequence`, `pad_packed_sequence`, `F`, `MaskedNorm` |
| `decoders.py:4` | `uneven_subsequent_mask`, `ConfigurationError` |
| `BiLSTM.py:1,4` | `pdb`, `F` |
| `discriminator_Data.py:4,5` | `Embeddings`, `FusionLayer` |
| `transformer_layers.py:7,8` | `np`, `F` |
| `initialization.py:11` | `Tensor` |
| `helpers.py:19,22,24` | `linear_sum_assignment`, `Dataset`, `Vocabulary` |
| `plot_videos.py:8` | `torch` |
| `__main__.py:4` | `os` |
| `tsn.py:1` | `json` |

### Codigo muerto

| Archivo | Codigo muerto |
|---------|---------------|
| `decoders.py:22-123` | `TransformerDecoder` — nunca se instancia |
| `loss.py:64-93` | `XentLoss` — nunca se usa |
| `search.py:greedy()` | Nunca se llama (comentado en model.py:200-207) |
| `CVT_training.py:598-643` | `_train_batch()` — nunca se invoca, referencia variables inexistentes |
| `model.py:280-282` | `dec_dropout`, `dec_emb_dropout`, `decoder_trg_trg` — variables computadas pero no usadas |
| `model.py:106` | `gloss_out = None` — definido pero nunca retornado |
| `tsn.py:25` | `new_shape` — computado pero no usado |

### Duplicacion

| Archivo | Issue |
|---------|-------|
| `encoders.py` vs `pre_encoders.py` | Codigo casi identico duplicado. `encoders.py` lo usa `model.py`, `pre_encoders.py` lo usa `Conv_model.py` |

### Otros

| Archivo | Issue |
|---------|-------|
| `__main__.py:23-24` | `--talent_ckpt` argumento aceptado pero nunca pasado a ninguna funcion |
| `BiLSTM.py:8-9` | Parametros `debug` y `num_classes` en constructor, nunca usados |
| `initialization.py:36,130` | Parametro `trg_padding_idx` aceptado pero nunca usado |
| `CVT_prediction.py:76` | `output = output` — no-op |
| `batch.py:14-20,37,67` | Variables definidas y sobreescritas o nunca usadas (`src_mask`, `nseqs`, `ntokens`, etc.) |

---

## Tabla Resumen

| Severidad | Cantidad | Mas criticos |
|-----------|----------|-------------|
| BUG | 20 | Return count mismatch en test (#1), future_prediction hardcodeado en batch.py (#2), decode() longitudes incorrectas (#4), checkpoint logic invertida (#5) |
| WARNING | 19 | HuberLoss ignora config (#21), hardcoded 32 en discriminador (#22), validacion solo 20 muestras (#28), gaussian_noise default contradictorio (#27) |
| CLEANUP | 24 | Dead imports en ~13 archivos, codigo muerto (TransformerDecoder, XentLoss, greedy, _train_batch), encoders duplicados |






Los más críticos son estos — causarían crash en runtime:

  1. CVT_prediction.py return count mismatch (BUG #1)(YA CORREGIDO)

  CVT_test en CVT_training.py:753-754 intenta desempaquetar 11 valores, pero
  pre_validate_on_data solo retorna 9. Esto significa que la inferencia/testing
  no funciona. Crashea con ValueError en cuanto corres python __main__.py --ckpt
  ... CVT_test ./Configs/Base.yaml.

  2. batch.py:29 — future_prediction = 10 hardcodeado (BUG #2)(NO CORREGIDO PORQUE NO SE CAMBIARA EL VALOR DE future_prediction = 10 )

  Si cambias future_prediction en el YAML a cualquier otro valor, Batch() sigue
  usando 10 internamente. El modelo recibiría datos con dimensiones
  inconsistentes — produce resultados silenciosamente incorrectos o crashea por
  shape mismatch. Esto es crítico porque no da error visible, simplemente entrena
   mal.

  3. CVT_training.py:151-159 — Checkpoint continuation logic invertida (BUG #5) (YA CORREGIDO)

  Cuando continue: True y no pasas --ckpt manualmente, el código encuentra el
  checkpoint pero no lo carga. El modelo arranca desde cero creyendo que
  continuó. Esto es crítico para entrenamientos largos que se interrumpen y
  reanudan.

  4. Conv_model.py:150 — //10 hardcodeado en AEdecode (BUG #7) (NO CORREGIDO PORQUE NO SE CAMBIARA EL VALOR DE future_prediction = 10 )

  Igual que #2: si future_prediction cambia de 10, el slicing
  skel_out.shape[2]//10 produce dimensiones incorrectas tanto en training como en
   inference.

  ---
  Los demás bugs son reales pero no tan urgentes porque:
  - Los de self.pre_model y _train_batch (#11-12) solo afectan código muerto que
  nunca se ejecuta
  - El gradient clipping (#13) afecta training pero no crashea
  - El decode() con longitudes incorrectas (#4) está en model.py que no es el
  Model activo (se usa Conv_model.py)
  - El train() vs CVT_train() (#6) está en un __main__ que nunca se ejecuta (se
  usa __main__.py del root)

  Resumen: los 4 críticos son #1, #2, #5, y #7. El #1 bloquea testing
  completamente. Los #2 y #7 están acoplados al mismo valor hardcodeado de
  future_prediction. El #5 puede causar pérdida de progreso de entrenamiento.



  Listo. Comenté 4 cosas en CVT_training.py:                                                  
                                                                                              
  1. Línea 33 — import de plot_video y alter_DTW_timing                                       
  2. Líneas 478-488 — llamada a produce_validation_video durante validación (val_inf)       
  3. Líneas 489-499 — llamada a produce_validation_video para talent                          
  4. Líneas 790-799 — llamada a produce_validation_video en CVT_test()                        
                                                                                              
  El método produce_validation_video sigue definido en la clase pero ya no se invoca.         
  Training, validación e inferencia funcionan igual — solo se deja de generar videos. 



Hay un crash garantizado en inferencia. En CVT_test línea 769:                                                                                                                                        
predicted_latent_data = [t.tolist() for t in predicted_latent]                                                                                                                                     
Pero pre_validate_on_data retorna None para predicted_latent (línea 128). Iterar sobre None lanzará un TypeError.
Se puede Proteger la línea 769 con un if (ya que el código que usa esos datos está comentado de todas formas)
En CVT_test, después del unpack (línea 756-757), el único código que usa predicted_latent es la línea 769:                                    
    predicted_latent_data = [t.tolist() for t in predicted_latent]                                                                                  
                                                                                                                                                
Y todo lo que sigue que usaría predicted_latent_data y reconst_latent ya está comentado (líneas 772-785). Entonces esa línea 769 hace un      
cálculo que nadie consume — pero crashea porque predicted_latent es None.                                                                                    
simplemente comentar o eliminar esa línea 769, ya que su resultado no se usa en ningún lado. El test correría sin problemas.
