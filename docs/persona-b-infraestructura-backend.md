# Persona B — Infraestructura y Backend

## Objetivo

Dejar reproducible la preparación técnica de la Entrega 1 del Sistema RAG NoSQL: conexión a MongoDB Atlas, colecciones con validación, índices requeridos y una API documentada con búsqueda textual, vectorial e híbrida, búsqueda de imágenes y respuestas RAG con Google Gemini. Los planes de las Personas A y C están en `docs/persona-a-analisis-y-modelado.md` y `docs/persona-c-dataset-y-chunking.md`.

## Qué quedó implementado

- Configuración mediante `.env` para URI de MongoDB, base de datos, índices vectoriales, modelos de embeddings de texto e imagen y Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL`). El archivo `.env` está excluido de Git.
- Script de prueba de conexión mediante `ping`.
- Inicializador idempotente de las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`.
- Validadores `$jsonSchema` con campos y tipos básicos. Los chunks requieren `doc_id`, `estrategia_chunking`, texto, fecha de ingesta y un vector de 384 componentes; además llevan `idioma` (copiado del artículo) y `parametros_chunking`. Los artículos admiten `url` y `licencia`; las imágenes, `clave`, `licencia`, `atribucion`, `modelo_embedding` y un vector de 512 componentes.
- Índice compuesto `{ fecha: 1, idioma: 1 }`, índice de texto en título y contenido con idioma por defecto español (stemming y stop words en español; si existía en otro idioma, el script lo recrea), e índice compuesto para localizar chunks por documento y estrategia.
- Solicitud de creación de dos índices de Atlas Vector Search: `chunks.embedding` (384 dimensiones, similitud coseno, filtros por `estrategia_chunking` e `idioma`) e `imagenes.embedding` (512 dimensiones, filtro por `categoria`), más un índice único por `imagenes.url`. Si el índice ya existía con otra definición, el script lo actualiza. Si M0 rechaza la operación desde el driver, el script imprime la definición para crearla desde la interfaz de Atlas.
- API FastAPI con `GET /health`, `POST /search`, `POST /rag` y documentación automática en `/docs`. La API arranca aunque MongoDB no responda; en ese caso `/health` devuelve `503` y se recupera sola cuando la conexión vuelve.
- `/search` con tres modos: `texto` (índice `$text`), `vector` (Atlas Vector Search sobre chunks, un resultado por artículo con su fragmento más parecido) e `hibrido` (por defecto; combina ambos rankings con Reciprocal Rank Fusion). Filtros por idioma, rango de fechas y estrategia de chunking.
- `/rag`: genera el embedding de la pregunta, recupera los chunks más parecidos con `$vectorSearch` (unidos a su artículo con `$lookup`) y pide a Gemini una respuesta basada solo en ese contexto, citando los fragmentos como `[1]`, `[2]`. Devuelve también las fuentes usadas. Si no hay chunks relevantes, responde sin llamar a Gemini.
- Búsqueda de imágenes con CLIP: `POST /imagenes/buscar` (por descripción en texto, con el codificador multilingüe alineado al de imágenes) y `POST /imagenes/similares` (por `imagen_id` o URL). Cada imagen incluye los artículos que la referencian, y `/rag` devuelve las imágenes de los artículos fuente (caso multimodal).
- Historial de usuario: si `/rag` recibe `usuario_id`, guarda la pregunta embebida en `usuarios.historial`, limitada a las 20 más recientes con `$slice`.
- Script de ingesta `backend/scripts/ingest.py`: valida artículos e imágenes desde JSON/JSONL (o re-procesa los artículos de Atlas con `--desde-mongo`), enlaza imágenes por `clave`, detecta duplicados, genera chunks `sentence-aware` y `semantic` con los parámetros del plan de C (`backend/app/chunking.py`), guarda cada chunk con su embedding normalizado y genera el embedding CLIP de las imágenes. Es idempotente: actualiza el artículo por `titulo` + `fuente`, la imagen por `url`, y reemplaza los chunks por estrategia.
- Datos de prueba en `data/articulos_ejemplo.json` para recorrer el flujo completo: ingesta → `/search` → `/rag`.
- Script `backend/scripts/comparar_estrategias.py` para el entregable C5: ejecuta las mismas consultas contra ambas estrategias y guarda JSON y CSV con lo recuperado, columnas para la rúbrica y estadísticas de chunks. Plantilla de consultas en `data/consultas_experimentales.json`.
- Pruebas automatizadas con pytest (`backend/tests`), con base de datos y modelos simulados: API, imágenes, estrategias de chunking, ingesta y comparación.

## Cómo ponerlo en marcha

1. En MongoDB Atlas, crea un cluster M0, crea un usuario de base de datos y autoriza la IP del equipo en Network Access. Para ejecutar el inicializador más de una vez, el usuario necesita el rol **Atlas admin** (o `dbAdmin` sobre la base): actualizar validadores de colecciones existentes usa `collMod`, que el rol de solo lectura/escritura no permite. Si falta el permiso, el script lo avisa y continúa.
2. En la raíz del proyecto, instala dependencias en un entorno virtual:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   ```

3. Copia `.env.example` como `.env` y coloca allí la URI copiada desde Atlas y la API key de Gemini (gratuita en Google AI Studio). No compartas ni subas ese archivo. Si la contraseña contiene símbolos reservados, codifícalos para URL.
4. Ejecuta y confirma la conexión:

   ```powershell
   python backend/scripts/check_connection.py
   ```

5. Inicializa las colecciones e índices:

   ```powershell
   python backend/scripts/init_db.py
   ```

   Luego carga artículos y genera sus chunks (la primera vez descarga el modelo de embeddings):

   ```powershell
   python backend/scripts/ingest.py data/articulos_ejemplo.json
   ```

6. Arranca la API y abre `/docs`:

   ```powershell
   uvicorn app.main:app --app-dir backend --reload
   ```

## Esquema e índices

| Colección | Uso | Índices/reglas principales |
|---|---|---|
| `articulos` | Artículos, metadatos e IDs de imágenes relacionadas | Campos obligatorios; compuesto fecha/idioma; texto en título y contenido (español) |
| `imagenes` | Referencias y metadatos de imágenes | URL y categoría requeridas; colección separada para no duplicar archivos grandes |
| `chunks` | Fragmentos y embeddings | `estrategia_chunking` requerido; `idioma` opcional; 384 dimensiones; índice por documento/estrategia; índice vectorial Atlas con filtros de estrategia e idioma |
| `usuarios` | Usuario e historial pequeño embebido | Historial como arreglo; ID de usuario único |

El modelo de 384 dimensiones corresponde al ejemplo `all-MiniLM-L6-v2` del enunciado. Si el equipo cambia el modelo, debe cambiar `EMBEDDING_MODEL` en `.env` y `VECTOR_DIMENSIONS` en `backend/app/config.py`, volver a ejecutar `init_db.py` y regenerar los embeddings de los chunks con el mismo modelo. Atlas Free permite hasta tres índices Search/Vector Search; este diseño requiere uno vectorial. Si M0 rechaza crearlo mediante el driver, usa **Search & Vector Search** en la interfaz y pega la definición que muestra el script.

## Probar la API

Con el servidor corriendo, `GET http://127.0.0.1:8000/health` debe responder estado `ok` cuando MongoDB está accesible y `503` cuando no lo está. `POST http://127.0.0.1:8000/search` recibe un JSON como:

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

El modo `texto` requiere artículos cargados y el índice de texto; los modos `vector` e `hibrido` requieren además chunks con embeddings y el índice vectorial en estado READY. Si falta el índice vectorial, la API responde `503` indicando qué revisar.

`POST http://127.0.0.1:8000/rag` recibe:

```json
{
  "question": "¿Qué medidas de sostenibilidad se anunciaron en 2024?",
  "idioma": "es",
  "estrategia_chunking": "sentence-aware",
  "limit": 5,
  "usuario_id": "estudiante-1"
}
```

Códigos de error: `503` si MongoDB, el índice vectorial o el modelo de embeddings no están disponibles; `502` si falta `GEMINI_API_KEY` o Gemini devuelve un error (por ejemplo, cuota agotada).

Para correr las pruebas: `pip install -r backend/requirements-dev.txt` y luego `pytest backend/tests`.

## Pendientes externos y siguiente fase

El cluster, el usuario, la contraseña y la lista de IP permitidas dependen de la cuenta Atlas del equipo; el repositorio no contiene esas credenciales y por eso la conexión real queda pendiente de ejecutar localmente. Falta cargar el conjunto de artículos real del proyecto con `ingest.py` (en el formato descrito en el README) y ejecutar `comparar_estrategias.py` con las consultas adaptadas por C. Las búsquedas de imágenes requieren que el dataset incluya imágenes accesibles (URL o archivo en el repositorio). Si se cambia `EMBEDDING_MODEL`, hay que regenerar los chunks con `ingest.py --desde-mongo`. El nombre del modelo de Gemini (`GEMINI_MODEL`) puede cambiar con el tiempo; si Google lo retira, basta con actualizarlo en `.env`.

## Coordinación con A y C

Acuerdos que B deja implementados y que A y C deben confirmar o pedir cambiar antes de la ingesta final:

| Tema | Qué dejó B | Qué falta confirmar |
|---|---|---|
| Contrato de datos (A2) | Campos y validadores descritos en el README (sección *Formato del dataset*) y en `backend/scripts/init_db.py`. | A congela nombres y tipos; cualquier cambio se refleja en `init_db.py` e `ingest.py`. |
| Dataset (C1) | `ingest.py` lee `data/<archivo>.json` con `articulos` e `imagenes`; las imágenes pueden ser URL o rutas relativas como `data/imagenes/...`. | C entrega el dataset en esa ruta y decide si las imágenes van como archivos en el repositorio o como URLs. |
| Chunking (C2, C3) | `backend/app/chunking.py` implementa `sentence-aware` (5 oraciones, solapamiento 1) y `semantic` (umbral 0.80, 2 a 8 oraciones) detrás de `chunk_text(texto, estrategia)`. | C revisa casos reales (abreviaturas, listas, inglés) y ajusta parámetros o reglas en ese módulo; la versión queda registrada en cada chunk. |
| Modelos | Texto: `all-MiniLM-L6-v2` (384). Imágenes: `clip-ViT-B-32` (512) y su codificador de texto multilingüe. | Todo el equipo usa estos modelos; consultas y corpus deben compartir modelo. |
| Experimento (C4, C5) | `comparar_estrategias.py` y la plantilla `data/consultas_experimentales.json`. | C adapta las diez consultas y define con A la rúbrica antes de ejecutar. |
| Índices (A3) | `chunks_vector_index`, `imagenes_vector_index`, texto y compuesto fecha/idioma. | Confirmar en Atlas que los índices vectoriales estén en estado READY antes de presentarlos como creados. |
