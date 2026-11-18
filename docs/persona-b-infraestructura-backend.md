# Persona B — Infraestructura y Backend

## Objetivo

Dejar reproducible la preparación técnica de la Entrega 1 del Sistema RAG NoSQL: conexión a MongoDB Atlas, colecciones con validación, índices requeridos y una API mínima documentada.

## Qué quedó implementado

- Configuración mediante `.env` para URI de MongoDB, base de datos e índice vectorial. El archivo `.env` está excluido de Git.
- Script de prueba de conexión mediante `ping`.
- Inicializador idempotente de las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`.
- Validadores `$jsonSchema` con campos y tipos básicos. Los chunks requieren `doc_id`, `estrategia_chunking`, texto, fecha de ingesta y un vector de 384 componentes.
- Índice compuesto `{ fecha: 1, idioma: 1 }`, índice de texto en título y contenido con idioma por defecto español (stemming y stop words en español; si existía en otro idioma, el script lo recrea), e índice compuesto para localizar chunks por documento y estrategia.
- Solicitud de creación de Atlas Vector Search sobre `chunks.embedding`, con 384 dimensiones, similitud coseno y filtro por estrategia. Si M0 rechaza la operación desde el driver, el script imprime la definición para crearla desde la interfaz de Atlas.
- API FastAPI con `GET /health`, `POST /search`, `POST /rag` y documentación automática en `/docs`. La API arranca aunque MongoDB no responda; en ese caso `/health` devuelve `503` y se recupera sola cuando la conexión vuelve.
- Búsqueda textual inicial con filtros opcionales por idioma y rango de fechas. `/rag` responde `501` con el alcance pendiente explicado.

## Cómo ponerlo en marcha

1. En MongoDB Atlas, crea un cluster M0, crea un usuario de base de datos y autoriza la IP del equipo en Network Access.
2. En la raíz del proyecto, instala dependencias en un entorno virtual:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   ```

3. Copia `.env.example` como `.env` y coloca allí la URI copiada desde Atlas. No compartas ni subas ese archivo. Si la contraseña contiene símbolos reservados, codifícalos para URL.
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
| `chunks` | Fragmentos y embeddings | `estrategia_chunking` requerido; 384 dimensiones; índice por documento/estrategia; índice vectorial Atlas |
| `usuarios` | Usuario e historial pequeño embebido | Historial como arreglo; ID de usuario único |

El modelo de 384 dimensiones corresponde al ejemplo `all-MiniLM-L6-v2` del enunciado. Si el equipo cambia el modelo, debe cambiar `VECTOR_DIMENSIONS` en `backend/scripts/init_db.py` y mantener consistente el índice vectorial. Atlas Free permite hasta tres índices Search/Vector Search; este diseño requiere uno vectorial. Si M0 rechaza crearlo mediante el driver, usa **Search & Vector Search** en la interfaz y pega la definición que muestra el script.

## Probar la API

Con el servidor corriendo, `GET http://127.0.0.1:8000/health` debe responder estado `ok` cuando MongoDB está accesible y `503` cuando no lo está. `POST http://127.0.0.1:8000/search` recibe un JSON como:

```json
{
  "query": "sostenibilidad ambiental",
  "idioma": "es",
  "fecha_desde": "2024-01-01T00:00:00Z",
  "fecha_hasta": "2024-12-31T23:59:59Z",
  "limit": 10
}
```

La búsqueda requiere documentos cargados y los índices de texto creados. `/rag` está expuesto para cumplir el contrato inicial de la API, pero la recuperación semántica y la llamada a un LLM se completarán en la implementación RAG.

## Pendientes externos y siguiente fase

El cluster, el usuario, la contraseña y la lista de IP permitidas dependen de la cuenta Atlas del equipo; el repositorio no contiene esas credenciales y por eso la conexión real queda pendiente de ejecutar localmente. Después, se debe conectar la ingesta y el generador de embeddings, completar búsqueda vectorial/híbrida en `/search`, y conectar recuperación de chunks más proveedor LLM en `/rag`.
