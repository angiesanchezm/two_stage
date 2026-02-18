# Revision del Paper

**"A two-stage sign language generation framework with self-supervised latent representation learning"**
Published in: Knowledge-Based Systems 333 (2026)

---

## Fortalezas

1. **Idea central clara y bien motivada**: El enfoque de dos etapas (SLAE + GlossMapper) con un espacio latente intermedio es una contribucion razonable frente a los metodos que predicen keypoints directamente.

2. **Physical Perception loss bien disenada**: La combinacion de inter-frame smoothness + multi-region (trunk/left-hand/right-hand con pesos 0.2/0.4/0.4) tiene sentido fisico y los ablation studies lo respaldan (Table 2).

3. **Resultados experimentales positivos**: Superan al baseline PTR en PHOENIX14T en todas las metricas, y en How2Sign en BLEU-1 a BLEU-4.

4. **User study incluido** (Table 6): Especialmente el score de Motion Fluency (4.90 vs. 3.40 del segundo mejor) valida la contribucion principal.

---

## Debilidades / Observaciones

### Metodologia

- La Eq. 6 (loss del SLAE) parece calcular la diferencia entre frames consecutivos `(x_i - x_{i+1})^2`, no MSE entre input y output como dice el texto. Hay una inconsistencia entre la formula y la descripcion.
- El paper dice que el decoder usa BiLSTM porque "avoids a lot of matrix operations" comparado con Transformer -- esto es debatible y no esta bien justificado cuantitativamente.

### Experimentos

- En How2Sign, el ROUGE score (8.09) es **menor** que PTR (9.95), PTR-AT (9.98), SIGNGAN (9.96) y ASL-SLP (10.10). El paper lo justifica vagamente como "flexibility and diversity of English expressions" -- esta justificacion es debil.
- La comparacion no incluye DTW como metrica en las tablas, a pesar de que el codigo usa DTW como metrica principal de evaluacion. Seria importante mostrar esa metrica.
- Falta de analisis estadistico -- no hay barras de error, intervalos de confianza, ni multiples ejecuciones reportadas.
- El user study con solo 10 participantes no-signantes es limitado. Los autores lo reconocen en la conclusion.

### Presentacion

- Fig. 1 dice "predictd" (typo -> "predicted").
- Section 3: "The aim this method" -> "The aim of this method".
- Las ecuaciones 2-4 y 8-10 son practicamente identicas (standard attention). Ocupan espacio sin aportar novedad.
- La referencia [30] aparece dos veces con contenido diferente: Zelinka et al. en Section 2.1 y "50 joints [30]" en Section 4.2. Revisar la numeracion.
- El "repository link" en el abstract esta pendiente de rellenar.

### Reproducibilidad

- El `requirements.txt` del repo no corresponde con las dependencias reales (lista paquetes del sistema como `Brlapi`, `cupshelpers`, `pygobject`). Necesita actualizarse con las dependencias reales de Python (torch, numpy, opencv-python, scipy, etc.).

---

## Resumen

El paper presenta una contribucion valida con el framework de dos etapas y la Physical Perception loss. Los resultados en PHOENIX14T son convincentes. Las principales areas de mejora son: (1) justificar/analizar el bajo ROUGE en How2Sign, (2) corregir la inconsistencia en Eq. 6, (3) corregir typos, y (4) actualizar el repositorio para reproducibilidad.
