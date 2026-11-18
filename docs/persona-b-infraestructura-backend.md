# Persona B — Infraestructura y Backend

## Objetivo

Dejar reproducible la preparación técnica de la Entrega 1 del Sistema RAG NoSQL: conexión a MongoDB Atlas, colecciones con validación, índices requeridos y una API documentada con búsqueda textual, vectorial e híbrida, y respuestas RAG con Google Gemini.

## Qué quedó implementado

- Configuración mediante `.env` para URI de MongoDB, base de datos, índice vectorial, modelo de embeddings y Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL`). El archivo `.env` está excluido de Git.
- Script de prueba de conexión mediante `ping`.
- Inicializador idempotente de las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`.
- Validadores `$jsonSchema` con campos y tipos básicos. Los chunks requieren `doc_id`, `estrategia_chunking`, texto, fecha de ingesta y un vector de 384 componentes; además pueden llevar `idioma` (copiado del artículo) para filtrar la recuperación.
- Índice compuesto `{ fecha: 1, idioma: 1 }`, índice de texto en título y contenido con idioma por defecto español (stemming y stop words en español; si existía en otro idioma, el script lo recrea), e índice compuesto para localizar chunks por documento y estrategia.
- Solicitud de creación de Atlas Vector Search sobre `chunks.embedding`, con 384 dimensiones, similitud coseno y filtros por `estrategia_chunking` e `idioma`. Si el índice ya existía con otra definición, el script lo actualiza. Si M0 rechaza la operación desde el driver, el script imprime la definición para crearla desde la interfaz de Atlas.
- API FastAPI con `GET /health`, `POST /search`, `POST /rag` y documentación automática en `/docs`. La API arranca aunque MongoDB no responda; en ese caso `/health` devuelve `503` y se recupera sola cuando la conexión vuelve.
- `/search` con tres modos: `texto` (índice `$text`), `vector` (Atlas Vector Search sobre chunks, un resultado por artículo con su fragmento más parecido) e `hibrido` (por defecto; combina ambos rankings con Reciprocal Rank Fusion). Filtros por idioma, rango de fechas y estrategia de chunking.
- `/rag`: genera el embedding de la pregunta, recupera los chunks más parecidos con `$vectorSearch` (unidos a su artículo con `$lookup`) y pide a Gemini una respuesta basada solo en ese contexto, citando los fragmentos como `[1]`, `[2]`. Devuelve también las fuentes usadas. Si no hay chunks relevantes, responde sin llamar a Gemini.
- Historial de usuario: si `/rag` recibe `usuario_id`, guarda la pregunta embebida en `usuarios.historial`, limitada a las 20 más recientes con `$slice`.
- Pruebas automatizadas con pytest (`backend/tests`), con base de datos y modelos simulados.

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

El cluster, el usuario, la contraseña y la lista de IP permitidas dependen de la cuenta Atlas del equipo; el repositorio no contiene esas credenciales y por eso la conexión real queda pendiente de ejecutar localmente. La búsqueda vectorial y `/rag` dependen de que la ingesta cargue chunks con embeddings generados con el mismo modelo configurado en `EMBEDDING_MODEL` (por defecto `all-MiniLM-L6-v2`, normalizados) y, si se quiere filtrar por idioma en el índice, con el campo `idioma` copiado del artículo. El nombre del modelo de Gemini (`GEMINI_MODEL`) puede cambiar con el tiempo; si Google lo retira, basta con actualizarlo en `.env`.
