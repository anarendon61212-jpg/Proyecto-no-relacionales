# Sistema RAG NoSQL con MongoDB - backend

Backend del Sistema RAG NoSQL (Persona B: Infraestructura y Backend): MongoDB Atlas, búsqueda textual, vectorial e híbrida, y respuestas RAG generadas con Google Gemini.

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
- `GEMINI_MODEL` y `EMBEDDING_MODEL`: opcionales; los valores del ejemplo sirven.

No subas `.env` al repositorio.

## Verificar e inicializar

```powershell
python backend/scripts/check_connection.py
python backend/scripts/init_db.py
```

El inicializador crea o actualiza las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`, aplica reglas de validación, crea los índices convencionales, compuesto y de texto (este último con idioma español), y solicita el índice vectorial de Atlas para `chunks.embedding` (384 dimensiones, similitud coseno, filtros por `estrategia_chunking` e `idioma`). Si Atlas M0 no permite crearlo desde el driver, el script imprime la definición JSON para crear el índice desde **Search & Vector Search** en Atlas. Atlas puede tardar un momento en dejar el índice listo.

## Ejecutar la API

```powershell
uvicorn app.main:app --app-dir backend --reload
```

- `GET /health`: comprueba la conexión a MongoDB (`503` si no está disponible; la API arranca de todos modos).
- `POST /search`: busca artículos en modo `texto`, `vector` o `hibrido` (por defecto), con filtros de idioma, fechas y estrategia de chunking.
- `POST /rag`: recupera los chunks más parecidos a la pregunta y genera con Gemini una respuesta que cita sus fuentes.
- `GET /docs`: documentación interactiva de FastAPI.

La primera búsqueda vectorial o pregunta RAG tarda unos segundos más porque se descarga y carga el modelo de embeddings; las siguientes son rápidas.

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

La respuesta incluye `respuesta` (con citas como `[1]`), `modelo` y `fuentes`: los chunks usados, con título, fuente, fecha, score y fragmento. Si se envía `usuario_id`, la pregunta se guarda en el historial del usuario (solo las 20 más recientes).

## Pruebas

```powershell
pip install -r backend/requirements-dev.txt
pytest backend/tests
```

Las pruebas usan una base de datos simulada; no necesitan Atlas, el modelo de embeddings ni la API key de Gemini.

## Notas de alcance

Crear el cluster, el usuario y la regla de IP se hace en la cuenta Atlas del equipo. La búsqueda vectorial necesita que la ingesta haya cargado chunks con embeddings del **mismo modelo** configurado en `EMBEDDING_MODEL` (por defecto `all-MiniLM-L6-v2`, 384 dimensiones).
