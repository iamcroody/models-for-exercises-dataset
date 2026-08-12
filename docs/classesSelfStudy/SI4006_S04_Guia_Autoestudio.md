# SI4006 · Semana 4 — Guía de autoestudio (sesión asincrónica)

**Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · Módulo 1 — Transformers**
**Tema:** Hugging Face, fine-tuning supervisado y **LoRA**

---

## Justificación

La sesión de esta semana cae en **festivo (7 de agosto, Batalla de Boyacá)**, así que **no hay clase presencial**, y a voto de la mayoría en clase se determinó subir el contenido y estudiarlo cada uno.

Probablemente les tome cerca de dos a tres horas ejecutarlo completo a consciencia.

El objetivo es que cada equipo tenga un **baseline fine-tuneado** de su modelo, corriendo en Colab gratis, y sepa **compararlo honestamente** (al menos inicialmente) contra un punto de partida.

---

## Ruta de la semana (háganla en orden)

1. **Lean esta guía** (secciones 1 a 4). Es la "clase" en texto.
2. **Corran el notebook** `S04_Lab_Fine-tuning_LoRA.ipynb` de principio a fin en Colab. Es un ejemplo completo y funcional de fine-tuning con LoRA que van a **copiar y adaptar** a su dominio.
3. **Consulten** `SI4006_Guia_Datos_y_Datos_Sinteticos.md` para armar el dataset de su proyecto.

---

## 1 · Hugging Face

El fine-tuning de esta guía se apoya en unas pocas librerías que trabajan juntas:

| Librería | Para qué sirve |
|---|---|
| `transformers` | Cargar modelos, tokenizers y la **Trainer API** (el loop de entrenamiento hecho). |
| `datasets` | Cargar, filtrar y preparar datos (splits train/val/test). |
| `peft` | **LoRA** y otros métodos de fine-tuning eficiente. |
| `evaluate` | Métricas (accuracy, F1, ROUGE…). |
| `accelerate` | Que el entrenamiento use GPU/CPU sin que ustedes lo cableen a mano. |

Ideas clave:
- **El Hub** (<https://huggingface.co>) es como GitHub pero de modelos, datasets y demos (*Spaces*). De ahí sacan su modelo base, y datasets para esta entrega, o cualquier otro ejercicio.
- Cargar un modelo o un tokenizer es **una línea** (`AutoModelForSequenceClassification.from_pretrained(...)`, `AutoTokenizer.from_pretrained(...)`).

> **Sobre paquetes en Colab (2026):** Colab ya trae `transformers` reciente (5.x) y `torch` 2.x. **No fijen versiones viejas** de `transformers` pues termina rompiendo otras cosas y obliga a reiniciar. Solo instalen lo que falte (`peft`, `datasets`, `evaluate`).

---

## 2 · Fine-tuning supervisado

El **Fine-tuning** se puede definir, de manera reducida, como tomar un modelo ya pre-entrenado (que ya "sabe" lenguaje) y **seguir entrenándolo un poco** con **datos de su tarea** para que la resuelva bien. Imaginenlo como aprender a derivar parcialmente luego de saber derivar, no vuelven a aprender todo desde la definición de límite, ya saben algunas cosas y las complementan o expanden y quedan sabiendo ambas cosas.

El flujo de este proceso, supervisado, suele ser el mismo:
```
se consiguen datos etiquetados (o se etiquetan)  ->  tokenizar  ->  Trainer (entrena)  ->  evaluar en validación
```

Y con **La Trainer API** ya tienen el loop de entrenamiento: no tienen que escribir el `for epoch...` a mano. Ustedes le dan el modelo, los datos tokenizados, la métrica y unos **hiperparámetros mínimos**:

| Hiperparámetro | Qué controla | Valor típico para empezar |
|---|---|---|
| `learning_rate` | qué tan grande es cada paso de aprendizaje | `2e-4` (con LoRA), `2e-5` (full) |
| `num_train_epochs` | cuántas pasadas a los datos | 1–3 |
| `per_device_train_batch_size` | cuántos ejemplos por paso | 8–16 (bájenlo si se queda sin memoria) |
| `weight_decay` | regularización (evita sobreajuste) | `0.01` |

> **⚠️ Error frecuente:** entrenar sobre **todo** el dataset sin separar `train` / `validation` / `test`. **Antes de tocar el modelo, hagan el split.** Si evalúan sobre los mismos datos con que entrenaron, no sirve.

---

## 3 · LoRA: fine-tuning eficiente (el corazón del módulo)

Un modelo grande tiene **millones o miles de millones** de pesos. Reentrenarlos **todos** (full fine-tuning) es caro y tenemos las capacidades computacionales para hacerlo. Pero **LoRA** (Low-Rank Adaptation) lo soluciona:

> En vez de mover todos los pesos, LoRA **"congela" el modelo original** y aprende **dos matrices pequeñas** (de bajo rango) que se suman al modelo. Entrenan **<1%** de los parámetros y obtienen casi el mismo resultado.

Adaptar un modelo a una tarea nueva suele requerir cambios "de baja dimensionalidad", no hace falta reescribir todo el modelo, solo empujarlo un poco en la dirección correcta.

**Los tres hiperparámetros de LoRA que deben saber nombrar (prinicplmente):**
- `r` (**rank**): el tamaño de las matrices pequeñas. Más grande = más capacidad, más costo. Típico: 8–16.
- `lora_alpha`: cuánto pesa el ajuste de LoRA. Típico: 16–32 (regla común: `alpha ≈ 2·r`).
- `target_modules`: **a qué capas** se les aplica LoRA (normalmente las de atención: `q_proj`, `v_proj`…). Cambia según la familia del modelo.

**LoRA vs. full fine-tuning, cuándo usar cuál:**
- **LoRA** (lo que usarán): modelos grandes, poca GPU, datasets pequeños/medianos. Es el estándar hoy.
- **Full fine-tuning**: solo si el modelo es muy pequeño y tienen muchos datos y GPU de sobra. **No ahora.**
- **QLoRA** = LoRA + el modelo base **cuantizado** (comprimido a 4 bits) para que quepan modelos aún más grandes. Misma idea, más ahorro de memoria.

Acá les dejo el artículo en ArXiv por si quieren mayor detalle, no es muy largo pero puede que unas partes sean algo teoricas fuertes, es bueno que le den una mirada.
https://arxiv.org/abs/2106.09685

---

## 4 · El notebook

Abran **`S04_Lab_Fine-tuning_LoRA.ipynb`** en Colab y córranlo de principio a fin. Es un ejemplo **completo y funcional**: fine-tunea un modelo pequeño con LoRA sobre un dataset real y lo **compara contra un baseline**. Cada sección está comentada.

Lo que el notebook les muestra, paso a paso:
1. Instalar solo lo que falta (sin romper Colab).
2. Cargar un dataset y hacer el **split** correcto.
3. Cargar modelo base + tokenizer y tokenizar los datos.
4. **Medir el baseline** (el modelo *sin* fine-tuning) — el número contra el que van a comparar.
5. Configurar **LoRA** con `peft` (`r`, `lora_alpha`, `target_modules`).
6. Entrenar con la **Trainer API**.
7. **Evaluar** el modelo afinado y ver el **delta** frente al baseline.
8. Mirar **ejemplos cualitativos** de entrada → salida.

> **Su misión:** ese notebook es el molde. **Cámbienlo** para usar el modelo base y el dataset de **su** proyecto (la familia que decidieron en la Semana 3). El notebook usa la ruta de **clasificación (encoder)** como ejemplo; al final tiene una sección **"Cómo adaptar esto a las otras dos familias"** (decoder y enc-dec) para los equipos que generan o transforman texto.

**Tip de Colab:** activen la GPU gratis en `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`. Con LoRA y un modelo pequeño, el entrenamiento del ejemplo toma pocos minutos.

---

## 5 · W&B (tracking) — opcional esta semana

En clase habríamos mostrado **Weights & Biases** en vivo para ver el entrenamiento graficado; es la herramienta de seguimiento del curso. Para esta semana **es opcional**: el notebook corre sin cuenta de W&B (`report_to="none"`). Si quieren adelantar, creen una cuenta gratis en <https://wandb.ai> y cambien esa línea a `report_to="wandb"`; el notebook lo indica.

---

## 6 · Autoevaluación

Respóndanse en equipo antes de dar la semana por vista:
- [ ] ¿Podemos explicar, en una frase, qué hace `peft` y por qué usamos LoRA en vez de full fine-tuning?
- [ ] ¿Sabemos decir qué son `r`, `lora_alpha` y `target_modules`?
- [ ] ¿Corrimos el notebook completo en Colab **sin errores**?
- [ ] ¿Tenemos claro por qué hace falta un **split** y un **baseline**?
- [ ] ¿Ya identificamos **el dataset** de nuestro proyecto (real, sintético o mezcla) usando la guía de datos?
- [ ] ¿Sabemos cuál es nuestra **métrica** principal y por qué?

Si marcaron todo, están listos para cerrar M1. Si algo quedó flojo, tráiganlo al canal o al inicio de la Semana 5.

---

## Materiales de esta semana
- `S04_Lab_Fine-tuning_LoRA.ipynb` — notebook-tutorial (córranlo).
- `SI4006_Guia_Datos_y_Datos_Sinteticos.md` — cómo conseguir/generar los datos.
- `STAI_M1_Asignacion_Fine-tuning_baseline.md` — la entrega, la rúbrica y el checklist.

*SI4006 · Universidad EAFIT · Semana 4 — Autoestudio.*
