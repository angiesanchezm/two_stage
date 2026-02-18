# Inconsistencias Codigo vs Paper

**Paper:** "A two-stage sign language generation framework with self-supervised latent representation learning"
**Repo:** SG_Latent

---

## CRITICAS

### 1. Discriminador/GAN existe en el codigo pero NO se menciona en el paper

El codigo en `CVT/CVT_training.py:349-388` tiene un loop completo de entrenamiento adversarial con `BCEWithLogitsLoss` y peso `0.0001`. El discriminador esta en `discriminator_Data.py` (`DataClassifierLayers`). Se instancia en `CVT/Conv_model.py:250-252`.

**Paper (Algorithm 1, linea 16):** `L_GlossMapper = L_SLAE + L_SL_GM + L_PhysicalPerception`

**Codigo (loss real):** `L_total = L_SL_GM + L_SLAE + L_multi-region + L_inter-frame + 0.0001 * L_disc`

El discriminador tiene su propio optimizer, se guarda/carga con checkpoints, y tiene su propia seccion en `Base.yaml:54-63`. Es una omision significativa del paper.

### 2. El codigo solo calcula DTW, NO BLEU/ROUGE

El paper reporta BLEU-1 a BLEU-4 y ROUGE como metricas de evaluacion. Sin embargo, `CVT/CVT_prediction.py:114-123` solo computa DTW (Dynamic Time Warping). No hay codigo de back-translation, BLEU ni ROUGE en ningun archivo del repositorio.

Config confirma (`Base.yaml:40-44`):
```yaml
early_stopping_metric: "dtw"
eval_metric: "dtw"
```

Los scores BLEU/ROUGE reportados en el paper no pueden reproducirse con este codigo.

---

## ALTAS

### 3. Multi-region loss usa HuberLoss, no MSE

**Paper (Eq. 13-17):** Physical Perception Loss usa MSE para ambos componentes.

**Codigo (`CVT_training.py:75-76`):**
```python
self.loss = torch.nn.MSELoss()        # inter-frame: OK, coincide
self.mulit_loss = torch.nn.HuberLoss() # multi-region: NO es MSE
```

HuberLoss se comporta como L1 para errores grandes y L2 para errores pequenos -- es una funcion de loss fundamentalmente diferente a MSE.

### 4. SLAE no se entrena por separado (self-supervised) -- es end-to-end

**Paper:** Stage 1 (SLAE) es un autoencoder "self-supervised" entrenado independientemente. Despues, Stage 2 (GlossMapper) usa el espacio latente del SLAE.

**Codigo:** No existe script ni modo separado para Stage 1. En `CVT_training.py:312-313`, el SLAE se ejecuta inline dentro del mismo training loop que el GlossMapper:
```python
latent_output,_ = self.model.AE(trg=batch[2])
latent_loss = self.loss(latent_output, batch[2])
```

Ambos stages se optimizan con el **mismo optimizer** en un solo backward pass (`CVT_training.py:363-367`).

### 5. Gradient clipping se aplica ANTES del backward pass (bug)

En `CVT_training.py:359-367`:
```python
if self.clip_grad_fun is not None:
    self.clip_grad_fun(params=self.model.parameters())  # <-- ANTES de backward

if update:
    self.optimizer.zero_grad()
    batch_loss.backward()
    self.optimizer.step()
```

El clipping se ejecuta sobre gradientes de la iteracion anterior (o gradientes stale). Lo estandar es clipear DESPUES de `backward()` y ANTES de `step()`. Esto hace que el gradient clipping sea no-funcional para el batch actual.

---

## MODERADAS

### 6. Conv1D: 2 capas, no 3

**Paper (Eq. 7):** "3 Conv1D layers (kernel 5x1)"

**Codigo (`Conv_model.py:56-65`):**
```python
self.temporal_conv = nn.Sequential(
    nn.Conv1d(src_length, trg_length, kernel_size=5, stride=1, padding=0),
    nn.BatchNorm1d(trg_length),
    nn.ReLU(inplace=True),
    nn.MaxPool1d(kernel_size=2, ceil_mode=False),
    nn.Conv1d(trg_length, trg_length, kernel_size=5, stride=1, padding=0),
    nn.BatchNorm1d(trg_length),
    nn.ReLU(inplace=True),
    nn.MaxPool1d(kernel_size=2, ceil_mode=False)
)
```

Son **2** Conv1D layers, no 3. Ademas incluyen BatchNorm + ReLU + MaxPool que el paper no menciona.

### 7. `src_encoder` se construye pero NUNCA se usa

`Conv_model.py:214-217` construye un TransformerEncoder:
```python
src_encoder = TransformerEncoder(**cfg["encoder"], ...)
```

Se almacena como `self.src_encoder` (linea 54) pero jamas se invoca en `forward()`, `AE()`, `AEencode()` ni `AEdecode()`. Son parametros muertos que consumen memoria pero nunca se entrenan.

### 8. GlossMapper reutiliza el encoder del SLAE en vez de tener su propio "Former"

**Paper:** Implica un "Former" (Transformer) separado para el GlossMapper.

**Codigo:** El `forward()` del GlossMapper en `Conv_model.py` usa `self.encoder` -- el mismo Transformer encoder que el SLAE usa en `AEencode()`. No hay un Transformer separado para el GlossMapper; reutiliza el del SLAE.

---

## BAJAS

### 9. Dimensiones hardcodeadas

En `Conv_model.py`:
- `src_length=18` (linea 240) -- max gloss length fijo
- `trg_length=102` (linea 242) -- max skeleton frames fijo
- `Linear(125, 512)` (linea 67) -- depende del output de temporal_conv
- `pose_time_dim=102` (linea 251) -- discriminador hardcodeado

Esto reduce la generalidad del modelo para otros datasets.

### 10. Default fallback de `learning_rate_min` del modelo no coincide con el config

En `Base.yaml` hay tres `learning_rate_min`:
- Linea 32: `training.learning_rate_min: 0.00002` (modelo/generator)
- Linea 57: `training.disc.learning_rate_min: 0.0002` (discriminador)
- Linea 62: `training.disc_hand.learning_rate_min: 0.0002` (discriminador de manos, no usado)

El valor del modelo (`0.00002`) es correcto y coincide con el paper. Sin embargo, en `CVT_training.py:84`:
```python
self.learning_rate_min = train_config.get("learning_rate_min", 0.0002)
```

El fallback default del `.get()` es `0.0002` (10x mayor que el config). Esto solo afectaria si alguien corriera el codigo sin esa key en su YAML. Con `Base.yaml` funciona correctamente.

### 11. Codigo muerto: `_train_batch()`

`CVT_training.py:598-643` define `_train_batch()` dentro de la clase `CVTTrainManager`, pero tiene dos problemas:

**Usa variables que nunca se definen en `__init__` (ni en ningun otro lugar):**
- `self.pre_model` (lineas 601, 625) -- no existe `self.pre_model = ...` en la clase
- `self.latent_loss` (linea 602) -- no existe `self.latent_loss = ...` en la clase
- `self.latent_optimizer` (lineas 631, 636) -- no existe `self.latent_optimizer = ...` en la clase

Si alguien llamara `_train_batch()`, Python crashearia con `AttributeError`.

**Nunca se invoca.** La unica referencia esta en la linea 393, **comentada**:
```python
# batch_loss, noise, latent = self._train_batch(batch, update=update)
```
El training loop real esta directamente inline en las lineas 312-388.

Ademas, `self.pre_model` tambien aparece en el bloque de Gaussian Noise (lineas 292-296), pero `gaussian_noise` esta en `False` en `Base.yaml`, asi que nunca se ejecuta. Si alguien lo activara, tambien crashearia.

**Que era probablemente:** `_train_batch()` parece ser de una version anterior del codigo donde el entrenamiento se hacia en dos fases reales: `self.pre_model` seria el SLAE pre-entrenado, `self.latent_optimizer` un optimizer separado para el SLAE, y `self.latent_loss` una loss function separada para el espacio latente. Esto coincide mas con lo que el paper describe (dos stages separados). La implementacion actual lo reemplazo con entrenamiento end-to-end inline, y el metodo viejo quedo sin borrar.

---

## Tabla Resumen

| # | Severidad | Problema | Paper dice | Codigo hace |
|---|-----------|----------|------------|-------------|
| 1 | CRITICA | Discriminador/GAN | No mencionado | Full GAN training (peso 0.0001) |
| 2 | CRITICA | Metricas evaluacion | BLEU-1 a BLEU-4 + ROUGE | Solo DTW |
| 3 | ALTA | Multi-region loss | MSE | HuberLoss |
| 4 | ALTA | Entrenamiento 2 stages | Self-supervised Stage 1, luego Stage 2 | Todo end-to-end, mismo optimizer |
| 5 | ALTA | Gradient clipping | Entrenamiento estandar | Clip ANTES de backward (bug) |
| 6 | MODERADA | Conv1D layers | 3 capas | 2 capas + BatchNorm + MaxPool |
| 7 | MODERADA | src_encoder | Parte de la arquitectura | Construido pero nunca usado |
| 8 | MODERADA | Former del GlossMapper | Transformer separado | Reutiliza el encoder del SLAE |
| 9 | BAJA | Dimensiones | Configurable | Hardcodeadas (18, 102, 125) |
| 10 | BAJA | lr_min fallback default | Config modelo: 0.00002 (correcto) | Fallback del .get() es 0.0002; solo afecta si falta la key en YAML |
| 11 | BAJA | Codigo muerto | N/A | _train_batch() con vars inexistentes |

---

## Anexo: Notas adicionales del analisis

### A1. Ubicacion del loss del GlossMapper

El loss completo se computa en `CVT/CVT_training.py` dentro del training loop:

- **Linea 312-313:** Loss SLAE (autoencoder reconstruction): `latent_loss = MSELoss(AE_output, batch[2])`
- **Linea 324-327:** Inter-frame loss: `frame_loss = MSELoss(diff_pred, diff_true)`
- **Linea 329-343:** Multi-region loss (HuberLoss): `mulit_loss = 0.2*body + 0.4*left_hand + 0.4*right_hand`
- **Linea 346:** Loss combinado: `batch_loss = MSE(skel_out, batch[2]) + latent_loss + mulit_loss + frame_loss`
- **Linea 349-358:** Se anade discriminador: `batch_loss += 0.0001 * disc_loss`

Loss real total:
```
L_total = L_SL_GM (MSE prediccion vs target)
        + L_SLAE (MSE reconstruccion autoencoder)
        + L_inter-frame (MSE diferencias entre frames)
        + L_multi-region (HuberLoss por regiones del cuerpo)
        + 0.0001 * L_disc (adversarial, no documentado en paper)
```

### A2. Data augmentation

Tres formas configuradas, solo una activa:

- **Gaussian Noise** (`model.py:72-75`): `gaussian_noise: False` en `Base.yaml` -- **desactivado**. El codigo que lo aplica esta comentado en `CVT_training.py:394-399`.
- **Future Prediction** (`batch.py:47-58`): **Activo** (`future_prediction: 10`). Para cada frame t, concatena keypoints de frames t a t+9, expandiendo el target de 150 a 1500 dims (+1 counter). Fuerza al modelo a aprender trayectorias temporales coherentes. Luego en `CVT_training.py:309` se reduce de vuelta a 151 dims. `batch[5]` (target expandido y reducido) se usa para inter-frame y multi-region loss; `batch[2]` (original) se usa para MSE principal y SLAE.
- **Frame Skipping** (`skip_frames: 1`): Valor 1 = sin skip. Si fuera >1, reduciria la resolucion temporal.

### A3. El discriminador: proposito y funcionamiento

`discriminator_Data.py` implementa `DataClassifierLayers`, un clasificador convolucional binario (real vs generado).

**Arquitectura:** `Linear(150->512)` + PositionalEncoding + 2x Conv1d + BatchNorm + ReLU + MaxPool + Linear(512->1)

Dentro del **mismo training loop** en `CVT/CVT_training.py:299-388`, hay dos pasos de optimizacion por iteracion:

1. **Paso 1 - Generator** (lineas 363-367): `self.optimizer` actualiza el modelo con todo el loss combinado (incluye `0.0001 * disc_loss` para enganar al discriminador)
2. **Paso 2 - Discriminador** (lineas 373-388): `self.disc_opt` actualiza solo el discriminador con `D_loss = 0.5 * (loss_real + loss_fake)` usando `BCEWithLogitsLoss`

Es entrenamiento adversarial GAN estandar. No requiere correr codigo separado. **El paper NO menciona este componente.**

### A4. `src_encoder`: construido pero nunca usado

En `build_model()` de `Conv_model.py` se crean **dos** TransformerEncoders:

| Variable | Linea | Asignado como | Uso real |
|----------|-------|---------------|----------|
| `src_encoder` | 214-217 | `self.src_encoder` (linea 54) | **NUNCA se llama** en ningun metodo |
| `encoder` | 231-233 | `self.encoder` (linea 78) | Usado por `forward()` Y `AEencode()` |

Verificacion -- `self.src_encoder` no aparece en:
- `forward()` (linea 84): usa `self.temporal_conv` + `self.encoder`
- `AE()` (linea 151): llama a `AEencode()` que usa `self.encoder`
- `AEencode()` (linea 126): usa `self.encoder`
- `AEdecode()` (linea 139): usa `self.decoder`

Probablemente `src_encoder` se construyo con la intencion de ser el "Former" separado del GlossMapper descrito en el paper, pero al final reutilizaron `self.encoder` (el del SLAE) para ambos caminos y nunca eliminaron el codigo. Son parametros muertos que consumen GPU.



Fixes realizados #9 y #10:
Fix hardcoded dimensions and learning_rate_min default

  - Replace hardcoded layer dimensions (125, 512, 1024) in Conv_model.py
    with values computed dynamically from config (embedding_dim, hidden_size)
  - Read src_length and trg_length from config YAML instead of hardcoding
    (18, 102) in build_model()
  - Use config hidden_size for discriminator instead of hardcoded 512
  - Fix learning_rate_min fallback default from 0.0002 to 0.00002 in
    CVT_training.py to match Base.yaml and paper