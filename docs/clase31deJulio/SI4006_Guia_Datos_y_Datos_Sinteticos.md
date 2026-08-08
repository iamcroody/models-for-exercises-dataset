# SI4006 · Guía práctica: cómo conseguir datos para su proyecto (y cómo generarlos con IA)

**Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT · Módulo 1**
Material de apoyo para la entrega **M1** (fine-tuning baseline) y para todo el proyecto.

---

## Justificación

En proyectos de IA, conseguir y preparar los datos es el 80% del trabajo real; el modelo es casi "fácil" en comparación. Como sus proyectos son de **tema libre y bastante específicos**, es muy probable que **no exista** un dataset listo para su caso y les toque **armarlo**: recolectando datos reales, generando datos sintéticos con ayuda de un modelo, o lo más común, **una mezcla de los dos**.

Esta guía les da un método para las **tres familias de problema** que vimos en la Semana 3, porque *qué* dato necesitan depende de *qué* hace su sistema:

| Si su sistema… | Familia (Semana 3) | Un "ejemplo" de datos es… |
|---|---|---|
| **clasifica / detecta / extrae** | Encoder (BERT) | `texto → etiqueta` (p. ej. reseña → `positiva/negativa`) |
| **genera / conversa / responde** | Decoder (Qwen, LLaMA) | `instrucción/pregunta → respuesta` |
| **transforma texto en texto** | Enc-dec (T5, BART) | `entrada → salida` (documento → resumen, texto → corregido) |

> **Regla de oro antes de tocar nada:** escriban **un par de ejemplos a mano** de su par `entrada → salida`. Si no logran escribir uno, todavía no tienen definida la tarea, y ningún dataset va a ser suficiente.

---

## Parte 1 · Primero conseguir datos REALES

Siempre que se pueda, **arranquen con datos reales**. Son la verdad de su dominio y son obligatorios para el conjunto de **prueba** (test): nunca evalúen la calidad final sobre datos inventados por un modelo. Fuentes, de la más fácil a la más laboriosa:

### 1.1 · Hugging Face Datasets (primer lugar donde buscar)
Miles de datasets abiertos, ya en formato listo para entrenar.
- Busquen en <https://huggingface.co/datasets> por tarea e idioma (filtros a la izquierda: *Task Categories*, *Languages*).
- Cargarlos es una línea:

```python
from datasets import load_dataset
# Ojo: el datasets nuevo exige el id CON namespace ('org/nombre'), ya no el alias corto.
ds = load_dataset("stanfordnlp/imdb")                    # clasificación de sentimiento (inglés)
ds = load_dataset("google-research-datasets/go_emotions") # clasificación multi-etiqueta de emociones
ds = load_dataset("csebuetnlp/xlsum", "spanish")         # resumen en español (enc-dec)
print(ds)                                  # miren splits, columnas y tamaños
print(ds["train"][0])                      # miren UN ejemplo real
```
> **Si les sale `HfUriError: Repository id must be 'namespace/name'`**, es justo esto: usaron el nombre corto (`imdb`) y el `datasets` nuevo pide el id completo (`stanfordnlp/imdb`). El id exacto está en la página del dataset en el Hub, arriba del todo.

- Filtren o recorten para que quepa en Colab gratis: `ds["train"].shuffle(seed=42).select(range(2000))`.

### 1.2 · Kaggle
Datasets de competencias y de la comunidad, muchos con dominios muy específicos (salud, finanzas, retail).
- <https://www.kaggle.com/datasets>. Descarguen el CSV y cárguenlo con `datasets` o `pandas`.
- **Revisen siempre la licencia** de cada dataset (columna *License* en Kaggle).

### 1.3 · Datos abiertos y APIs públicas
- Portales de datos abiertos (p. ej. **datos.gov.co**), APIs de noticias, Wikipedia, Reddit/PushShift, foros, catálogos de productos.
- Ventaja: son **de su dominio real**. Costo: casi siempre vienen **sin etiqueta** y toca limpiarlos.

### 1.4 · Scraping (con cabeza)
Si el dato vive en una página web, se puede extraer con `requests` + `BeautifulSoup` o `trafilatura`. **Antes de hacerlo:**
- Revisen los **términos de uso** del sitio y su `robots.txt`.
- **No** extraigan datos personales ni contenido tras login/paywall.
- Sean amables con el servidor (pausas entre peticiones, no lo saturen).
- Guarden **de dónde** salió cada dato: lo van a necesitar para la sección *Fuente* del M1.
- Sean **MUY CUIDADOSOS** con la trata de datos y las normativas al respecto.

### 1.5 · Datos propios / anotación manual
A veces la mejor opción es **etiquetar ustedes mismos** 100–300 ejemplos reales. Es más de lo que parece:
- Con **200–500 ejemplos reales bien etiquetados** ya se hace un fine-tuning decente con LoRA.
- Escriban una **guía de anotación** de media página (qué es cada etiqueta, casos de borde). Si dos personas del equipo etiquetan el mismo ejemplo distinto, su tarea está mal definida.

> **Licencia, privacidad y sesgo (no opcional en M1):** anoten la **licencia** de cada fuente, **quiten datos personales** (nombres, cédulas, correos) y escriban al menos **un sesgo o limitación** conocido de sus datos (idioma, región, periodo, quién quedó fuera).

---

## Parte 2 · Generar datos SINTÉTICOS con IA

Cuando no hay datos reales suficientes, se pueden **generar ejemplos con un modelo de IA**. Es legítimo y muy usado en la industria, pero tiene reglas.

### 2.1 · ¿Cuándo tiene sentido? ¿Cuándo no?
**Sí conviene** para: arrancar rápido cuando tienen 0 datos, **balancear** una clase minoritaria, cubrir casos raros a propósito, o **aumentar** (data augmentation) un set pequeño real.

**Tengan cuidado con:**
- **Sesgo heredado.** El modelo generador impone su propio estilo y sus sesgos; sus datos sintéticos se parecerán a él, no necesariamente a su dominio real.
- **Poca diversidad / "colapso".** Si piden 500 ejemplos con el mismo prompt, salen 500 casi iguales. Hay que forzar variedad (ver 2.3).
- **Fuga y evaluación tramposa.** **Nunca** midan la calidad final del modelo sobre datos sintéticos: el conjunto de **test siempre debe ser real** (o al menos revisado a mano). Si entrenan y evalúan con datos del mismo generador, el número miente.
- **No es oro puro.** Traten los datos sintéticos como un borrador: hay que **filtrarlos y revisar una muestra a mano**.

### 2.2 · El patrón general (sirve para las tres familias)

```
1. Definan el formato de salida EXACTO (p. ej. JSONL con campos {"texto", "etiqueta"}).
2. Escriban un prompt de generación con: rol + tarea + formato + 2-3 ejemplos (few-shot).
3. Generen en lotes, variando algo cada lote (tema, tono, dificultad, región) -> diversidad.
4. Parseen a JSONL/CSV.
5. LIMPIEN: quiten duplicados, quiten los que no cumplan el formato, revisen ~20 a mano.
6. Mezclen con datos reales y reserven un test REAL aparte.
```

**Dos formas de correr el generador (elijan según su acceso):**
- **Modelo local y gratis en Colab** (recomendado, sin API key): un decoder pequeño como `Qwen/Qwen2.5-1.5B-Instruct` vía `transformers`. Más lento y algo menos pulido, pero gratis y reproducible.
- **Modelo por API** (Claude, GPT u otro): más calidad y velocidad, pero necesita cuenta/clave y cuesta. Si lo usan, **revisen los términos de uso** sobre datos generados y **no manden datos privados** a la API.

### 2.3 · Receta por tipo de problema

#### A) Clasificación — familia **Encoder** (`texto → etiqueta`)
Objetivo: generar ejemplos de texto para cada clase, **balanceados** y **variados**.

- Generen **por clase** (un lote de "positivas", otro de "negativas", etc.) para controlar el balance.
- Fuercen variedad pidiendo distintos **temas, tonos, longitudes y registros** en cada lote.
- Formato de salida: una línea de JSON por ejemplo, `{"texto": "...", "etiqueta": "positiva"}`.

Prompt de ejemplo (para reseñas de una app, clase "negativa"):
```
Eres un generador de datos para entrenar un clasificador de sentimiento de reseñas
de una app de transporte en español de Colombia.
Genera 10 reseñas NEGATIVAS, variadas (distintos motivos: precio, demoras, la app falla,
mal trato), de 1 a 3 frases, en español coloquial.
Devuelve SOLO JSON, una línea por reseña: {"texto": "...", "etiqueta": "negativa"}
```

#### B) Generación — familia **Decoder** (`instrucción → respuesta`)
Objetivo: pares de **instrucción y respuesta** (esto se llama *SFT*, supervised fine-tuning).

- Primero generen una lista variada de **instrucciones/preguntas** que un usuario real haría de su dominio.
- Luego, para cada instrucción, generen una **respuesta buena** (idealmente con un modelo fuerte).
- Formato: `{"instruccion": "...", "respuesta": "..."}` (o el formato de chat que use su modelo).

Truco (**self-instruct**): denle 5–10 ejemplos "semilla" escritos por ustedes y pídanle que **genere instrucciones nuevas parecidas pero distintas**; así crece el set sin repetir.
```
Aquí tienes 5 preguntas típicas que un paciente le haría a un asistente de una clínica dental:
[sus 5 ejemplos]
Genera 15 preguntas NUEVAS, realistas y variadas, que NO repitan las anteriores.
Devuelve una pregunta por línea.
```

#### C) Transformación — familia **Enc-dec** (`entrada → salida`)
Objetivo: pares `entrada → salida` para resumir, reescribir, corregir, formalizar, traducir.

- Si ya tienen textos reales de entrada (documentos, mensajes), pídanle al modelo **solo la salida** (p. ej. el resumen); así la entrada es real y solo la etiqueta es sintética. Es lo mejor de los dos mundos.
- **Back-translation** para aumentar: traduzcan a otro idioma y de vuelta para obtener una paráfrasis natural de la misma frase.
- Formato: `{"entrada": "...", "salida": "..."}`.

```
Resume el siguiente reporte clínico en máximo 2 frases, en lenguaje sencillo para el paciente.
Reporte: <<pegan aquí un reporte REAL>>
Devuelve SOLO el resumen.
```

### 2.4 · Mini-receta ejecutable (generador local gratis en Colab)

```python
from transformers import pipeline

gen = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

def generar(prompt, n_lotes=5):
    salidas = []
    for i in range(n_lotes):
        msg = [{"role": "user", "content": prompt}]
        out = gen(msg, max_new_tokens=400, do_sample=True,
                  temperature=0.9, top_p=0.95)          # temperatura alta = más variedad
        salidas.append(out[0]["generated_text"][-1]["content"])
    return salidas

textos = generar("Genera 10 reseñas NEGATIVAS...  Devuelve SOLO JSON, una por línea.")
```
Luego parseen cada línea con `json.loads`, descarten las que fallen, quiten duplicados
(`set` sobre el texto) y guarden con `datasets` o escribiendo un `.jsonl`.

> **Diversidad:** suban `temperature` (0.8–1.1), cambien el tema/tono en cada lote y **dedupliquen**. Un set de 300 ejemplos variados vale más que 2000 casi idénticos.

---

## Parte 3 · Control de calidad (lo que separa un buen dataset de uno inútil)

1. **Deduplicar.** Quiten ejemplos idénticos o casi idénticos.
2. **Validar formato.** Descarten lo que no cumpla su esquema (`etiqueta` fuera de la lista, JSON roto).
3. **Revisar a mano una muestra.** Lean ~20–30 ejemplos ustedes mismos. Si a ustedes no les convencen, al modelo tampoco.
4. **Balancear las clases** (en clasificación).
5. **Separar los splits sin fuga:** `train` / `validation` / **`test` real**. Que un mismo texto no caiga en dos splits.
6. **Documentar el origen** de cada parte: cuánto es real, cuánto es sintético, con qué modelo y qué prompt lo generaron. Esto va directo a la descripción del dataset de M1.


*SI4006 · Universidad EAFIT · Módulo 1 — Guía de datos.*
