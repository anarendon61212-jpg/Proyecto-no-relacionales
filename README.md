# Sistema RAG NoSQL con MongoDB - backend

Backend del Sistema RAG NoSQL (Persona B: Infraestructura y Backend): MongoDB Atlas, ingesta con chunking, búsqueda textual, vectorial e híbrida, búsqueda de imágenes con CLIP y respuestas RAG generadas con Google Gemini.

Los planes de cada integrante están en `docs/`: Persona A (análisis y modelado), Persona B (infraestructura y backend) y Persona C (dataset y chunking).

## Requisitos

- Python 3.10 o superior.
- Un cluster MongoDB Atlas y un usuario de base de datos con rol **Atlas admin** (o `dbAdmin` sobre la base) para poder reejecutar el inicializador.
- La IP desde la que se ejecutará el backend autorizada en Network Access de Atlas.
- Una API key de Google Gemini (gratuita en [Google AI Studio](https://aistudio.google.com/apikey)) para usar `/rag`.

## Preparación

Desde la raíz del repositorio, crea y activa un entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

La primera instalación tarda varios minutos porque `sentence-transformers` descarga PyTorch.

Copia `.env.example` a `.env` y completa:

- `MONGODB_URI`: la cadena de conexión de Atlas. Si tu contraseña tiene caracteres reservados en una URL, codifícala antes de usarla.
- `GEMINI_API_KEY`: la API key de Gemini. Sin ella la API funciona, pero `/rag` responde `502`.
- `GEMINI_MODEL`, `EMBEDDING_MODEL`, `IMAGE_MODEL` e `IMAGE_TEXT_MODEL`: opcionales; los valores del ejemplo sirven.

No subas `.env` al repositorio.

## Verificar e inicializar

```powershell
python backend/scripts/check_connection.py
python backend/scripts/init_db.py
```

El inicializador crea o actualiza las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`, aplica reglas de validación, crea los índices convencionales, compuesto y de texto (este último con idioma español), y solicita dos índices vectoriales de Atlas: `chunks_vector_index` sobre `chunks.embedding` (384 dimensiones, similitud coseno, filtros por `estrategia_chunking` e `idioma`) e `imagenes_vector_index` sobre `imagenes.embedding` (512 dimensiones, filtro por `categoria`). Atlas Free permite hasta tres índices de este tipo. Si Atlas M0 no permite crearlos desde el driver, el script imprime la definición JSON para crearlos desde **Search & Vector Search** en Atlas. Atlas puede tardar un momento en dejar el índice listo.

## Cargar artículos y generar chunks

```powershell
python backend/scripts/ingest.py data/articulos_ejemplo.json
```

El script de ingesta valida cada artículo, lo guarda en `articulos` (y sus imágenes en `imagenes`), lo divide en chunks con las estrategias `sentence-aware` y `semantic` y guarda cada chunk con su embedding en `chunks`. Después genera el embedding CLIP de las imágenes que aún no lo tienen. Se puede ejecutar varias veces: un artículo se identifica por `titulo` + `fuente`, una imagen por su `url`, y los chunks se reemplazan, no se duplican.

- `data/articulos_ejemplo.json` trae 4 artículos de prueba (3 en español y 1 en inglés) para probar la API de punta a punta.
- Acepta un `.json` con una lista de artículos, un `.json` con `{"articulos": [...], "imagenes": [...]}` o un `.jsonl` con un artículo por línea.
- `--estrategias sentence-aware fixed-size` elige las estrategias; `--limite 10` procesa solo los primeros artículos.
- `--sin-imagenes` omite los embeddings de imágenes; `--reprocesar-imagenes` los regenera todos.
- `--desde-mongo` vuelve a generar los chunks de los artículos que ya están en Atlas (por ejemplo, después de cambiar de modelo de embeddings).

### Formato del dataset

| Campo | Artículo | Imagen |
|---|---|---|
| Obligatorios | `titulo`, `contenido_texto`, `idioma` (`es`, `en`…), `fecha` (`2024-05-01` o ISO 8601), `fuente` | `url`, `categoria` |
| Opcionales | `categoria`, `url`, `licencia`, `imagenes` | `clave`, `descripcion`, `fuente`, `licencia`, `atribucion` |

La `url` de una imagen puede ser `https://...` o una ruta **relativa a la raíz del repositorio** (por ejemplo `data/imagenes/panel.jpg`), nunca una ruta absoluta de un computador. Un artículo enlaza sus imágenes en `imagenes`, como objetos completos o como la `clave` de una imagen de la lista `imagenes`:

```json
{
  "articulos": [
    {
      "titulo": "Paneles solares en colegios",
      "contenido_texto": "...",
      "idioma": "es",
      "fecha": "2024-05-01",
      "fuente": "Nombre del medio",
      "categoria": "energía solar",
      "url": "https://medio.example/articulo",
      "licencia": "CC BY 4.0",
      "imagenes": ["panel-colegio"]
    }
  ],
  "imagenes": [
    {
      "clave": "panel-colegio",
      "url": "data/imagenes/panel-colegio.jpg",
      "categoria": "energía solar",
      "descripcion": "Paneles solares en el techo de un colegio",
      "fuente": "Wikimedia Commons",
      "licencia": "CC BY-SA 4.0",
      "atribucion": "Autor de la foto"
    }
  ]
}
```

Las imágenes de la lista se cargan aunque ningún artículo las referencie. Un artículo duplicado (mismo título y fuente) o con una clave de imagen inexistente se omite con un aviso.

### Estrategias de chunking

Parámetros del plan de la Persona C (`backend/app/chunking.py`, versión 2.0). Cada chunk guarda su estrategia y sus parámetros en `parametros_chunking`.

| Estrategia | Cómo divide el texto |
|---|---|
| `sentence-aware` | Hasta 5 oraciones por chunk; la última oración se repite al inicio del siguiente (solapamiento de 1). |
| `semantic` | Calcula el embedding de cada oración y corta cuando la similitud coseno entre dos oraciones consecutivas es menor que 0.80; cada chunk tiene entre 2 y 8 oraciones. |
| `fixed-size` | Ventanas de 150 palabras con 30 de solapamiento. No forma parte del experimento; queda disponible como referencia. |

Todas tienen un tope de 150 palabras para no exceder lo que `all-MiniLM-L6-v2` procesa sin truncar (256 tokens). Los títulos y elementos de listas se separan por línea, y un texto sin puntuación se divide en ventanas. El umbral 0.80 es el valor inicial del enunciado: con este modelo, oraciones consecutivas suelen tener similitudes menores, así que es probable que `semantic` corte a menudo y que el mínimo de 2 oraciones sea el que define muchos chunks. La comparación debe mostrar si conviene ajustarlo.

## Ejecutar la API

```powershell
uvicorn app.main:app --app-dir backend --reload
```

- `GET /health`: comprueba la conexión a MongoDB (`503` si no está disponible; la API arranca de todos modos).
- `POST /search`: busca artículos en modo `texto`, `vector` o `hibrido` (por defecto), con filtros de idioma, fechas y estrategia de chunking.
- `POST /rag`: recupera los chunks más parecidos a la pregunta y genera con Gemini una respuesta que cita sus fuentes. Incluye las imágenes de los artículos usados.
- `POST /imagenes/buscar`: imágenes que coinciden con una descripción en texto (español o inglés), con filtro opcional por `categoria`.
- `POST /imagenes/similares`: imágenes visualmente parecidas a una ya guardada (`imagen_id`) o a una URL `http(s)`.
- Cada imagen devuelta incluye los artículos que la referencian.
- `GET /docs`: documentación interactiva de FastAPI.

La primera búsqueda vectorial, pregunta RAG o búsqueda de imágenes tarda unos segundos más porque se descarga y carga el modelo correspondiente; las siguientes son rápidas.

Ejemplo de `POST /search`:

```json
{
  "query": "sostenibilidad ambiental",
  "modo": "hibrido",
  "idioma": "es",
  "fecha_desde": "2024-01-01T00:00:00Z",
  "fecha_hasta": "2024-12-31T23:59:59Z",
  "limit": 10
}
```

Ejemplo de `POST /rag`:

```json
{
  "question": "¿Qué medidas de sostenibilidad se anunciaron en 2024?",
  "idioma": "es",
  "estrategia_chunking": "sentence-aware",
  "limit": 5,
  "usuario_id": "estudiante-1"
}
```

Si no se indica `estrategia_chunking`, `/rag` usa `sentence-aware`, para no mezclar fragmentos repetidos de dos estrategias; conviene cambiarlo a la que gane la comparación. La respuesta incluye `respuesta` (con citas como `[1]`), `modelo`, `fuentes` (los chunks usados, con título, fuente, fecha, score y fragmento) e `imagenes` (las imágenes enlazadas a esos artículos). Si se envía `usuario_id`, la pregunta se guarda en el historial del usuario (solo las 20 más recientes).

Ejemplos de imágenes:

```json
{ "texto": "turbinas eólicas en el mar", "categoria": "energía eólica", "limit": 8 }
```

```json
{ "imagen_id": "665f1c2e8b3a4d0012345678", "limit": 8 }
```

## Comparar estrategias de chunking

```powershell
python backend/scripts/comparar_estrategias.py --limite 5
```

Ejecuta las consultas de `data/consultas_experimentales.json` contra los chunks `sentence-aware` y `semantic` con las mismas condiciones (mismo embedding de la consulta, filtros, límite y modelo). La plantilla trae las diez preguntas semilla del plan de la Persona C: hay que adaptarlas al dataset real **antes** de ejecutar y, si se puede, llenar `relevantes` con los títulos de los artículos esperados.

Guarda en `resultados/`:

- `comparacion_<fecha>.json`: lo recuperado por consulta y estrategia (y la respuesta de Gemini con `--con-respuesta`).
- `comparacion_<fecha>.csv`: una fila por chunk recuperado, con columnas vacías `relevancia`, `cobertura` y `precision` para calificar con la rúbrica.
- `estadisticas_<fecha>.csv`: chunks por estrategia, chunks por artículo y longitud promedio, mínima y máxima en palabras.

Si una consulta tiene `relevantes`, se calcula la precisión en k. Las consultas sin chunks que cumplan los filtros quedan marcadas como "sin evidencia".

## Pruebas

```powershell
pip install -r backend/requirements-dev.txt
pytest backend/tests
```

Las pruebas usan una base de datos y modelos simulados; no necesitan Atlas, el modelo de embeddings ni la API key de Gemini. Cubren la API (incluidas las imágenes), las estrategias de chunking, la ingesta y la comparación de estrategias.

## Notas de alcance

Crear el cluster, el usuario y la regla de IP se hace en la cuenta Atlas del equipo. La búsqueda vectorial compara la consulta con los chunks usando el modelo configurado en `EMBEDDING_MODEL` (por defecto `all-MiniLM-L6-v2`, 384 dimensiones); si se cambia el modelo, hay que regenerar los chunks con `python backend/scripts/ingest.py --desde-mongo`. Las imágenes usan CLIP (`clip-ViT-B-32`, 512 dimensiones) y las consultas de texto sobre imágenes usan su versión multilingüe alineada; los embeddings de texto y de imagen no son intercambiables y tienen índices separados.
