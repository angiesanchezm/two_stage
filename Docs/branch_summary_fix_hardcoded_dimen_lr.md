# Branch Summary: `fix/harcoded_dimen_lr`

## Resumen general (19 commits sobre `main`)

### 1. Setup inicial y documentacion

- `387d708` - Script de conversion de datos SignJoey al formato two_stage, limpieza de datos alemanes originales
- `a9161a6` - Agregar paper y documentacion .md
- `ccc298b` - Fix del .gitignore

### 2. Correccion de bugs criticos del codigo original

- `7a57523` - **Fix de dimensiones hardcodeadas** en `Conv_model.py` (125, 512, 1024 -> dinamicas desde config) y fix del `learning_rate_min` default
- `d3618a2` - Fix de mismatch en return values de `CVT_prediction` y logica de checkpoints
- `64e2090` / `aa99e1f` - Documentar issues y comentar generacion de video (no necesaria)
- `8c05a50` - **Adaptar modelo de 50 a 61 keypoints** (150 -> 183 coords)
- `89affac` - Agregar columna counter a los .skels (183 -> 184 valores/frame)
- `9fc9d2f` - Fix de unpack mismatch en validacion
- `7fa1a0c` - Eliminar `os.listdir("German")` hardcodeado que crasheaba
- `0f797d8` / `980b6e3` - Limpieza (.DS_Store, organizar docs)

### 3. Entrenamiento y evaluacion

- `4c8c945` - Fix de evaluacion truncada (solo 1 batch) y guardar resultados de inferencia a disco
- `0666b04` - **Aumentar max_length de 100 a 300 frames** (~97% del dataset)
- `0926478` - Documentar plan de reentrenamiento
- `bd50940` - **Fix BatchNorm crash** con ultimo batch incompleto (`drop_last=True`)

### 4. Post-procesamiento (lo mas reciente)

- `bb57c03` - Script `build_pt_gz_from_inference.py` para armar `.pt.gz` desde resultados de inferencia
- `5ea32dc` - Visualizador 3D interactivo para comparar predicciones vs ground truth
- `05b1dc3` **(ultimo commit)** - Flag `--trim` en `build_pt_gz` para restaurar las longitudes originales de las secuencias (quita padding)

## Detalle del ultimo commit: `--trim` flag (`05b1dc3`)

El commit mas reciente agrega la capacidad de recortar el padding de las secuencias generadas por inferencia. Las salidas del modelo estan paddeadas a longitud fija (302 o 102 frames), y el flag `--trim` carga las longitudes originales desde los archivos `.skels` de ground truth para quitar el padding izquierdo (6 frames) y devolver cada secuencia a su longitud real.

## Archivos principales modificados

| Archivo | Cambio |
|---|---|
| `CVT/Conv_model.py` | Dimensiones dinamicas desde config |
| `CVT/CVT_training.py` | Fix learning_rate_min, drop_last, max_length=300 |
| `CVT/CVT_prediction.py` | Fix return values, guardar resultados |
| `Configs/Base.yaml` | trg_size=183, trg_length=302, batch_size=32 |
| `data.py` | trg_size desde config en vez de hardcodeado |
| `helpers.py` | Fix menor |
| `convert_data_sidd_twostage/convert_signjoey_to_twostage.py` | Conversion de datos con counter |
| `convert_data_sidd_twostage/build_pt_gz_from_inference.py` | Construir .pt.gz + --trim |
| `convert_data_sidd_twostage/visualize.py` | Visualizador 3D interactivo |

---

En resumen lo que se hizo en esta rama fue:

  1. Adaptar el código original del paper a datos propios (SignJoey)

  El código original estaba diseñado para datos alemanes (PHOENIX) con
  50 keypoints. Se adaptó para trabajar con datos de SignJoey que
  tienen 61 keypoints (183 coordenadas). Esto implicó:

  - Crear el script de conversión (convert_signjoey_to_twostage.py)
  para transformar los datos al formato que espera el modelo
  - Cambiar trg_size de 150 a 183 en el config
  - Actualizar los índices de las regiones del cuerpo (body,
  right_hand, left_hand) en la función de loss

  2. Arreglar bugs y código hardcodeado del repo original

  El código original tenía muchas cosas hardcodeadas que impedían
  usarlo con otros datos:
  - Dimensiones fijas (125, 512, 1024) en el modelo → ahora se calculan
   desde el config
  - os.listdir("German") que crasheaba si no existía esa carpeta
  - trg_size=151 hardcodeado en data.py
  - learning_rate_min con valor incorrecto
  - Mismatch en return values que causaban crashes

  3. Escalar a secuencias más largas

  Los datos nuevos tienen secuencias más largas que las alemanas, así
  que se aumentó max_length de 100 a 300 frames.

  4. Herramientas de post-procesamiento

  Para poder evaluar los resultados con el pipeline de SignJoey, se
  crearon scripts para:
  - Construir .pt.gz desde las salidas de inferencia
  - Visualizar predicciones vs ground truth en 3D
  - Recortar el padding para obtener las longitudes reales

  En esencia: tomar el modelo del paper, hacerlo funcionar con tus
  propios datos, y conectar su salida con tu pipeline de evaluación.


