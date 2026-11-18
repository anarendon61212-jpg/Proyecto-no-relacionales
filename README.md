# Sistema RAG NoSQL con MongoDB - backend inicial

Base de trabajo para la Entrega 1 (Persona B: Infraestructura y Backend).

## Requisitos

- Python 3.10 o superior.
- Un cluster MongoDB Atlas y un usuario de base de datos.
- La IP desde la que se ejecutará el backend autorizada en Network Access de Atlas.

## Preparación

Desde la raíz del repositorio, crea y activa un entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Copia `.env.example` a `.env` y reemplaza `MONGODB_URI` con la cadena de conexión de Atlas. No subas `.env` al repositorio. Si tu contraseña tiene caracteres reservados en una URL, codifícala antes de usarla.

## Verificar e inicializar

```powershell
python backend/scripts/check_connection.py
python backend/scripts/init_db.py
```

El inicializador crea o actualiza las colecciones `articulos`, `imagenes`, `chunks` y `usuarios`, aplica reglas de validación, crea los índices convencionales, compuesto y de texto (este último con idioma español), y solicita el índice vectorial de Atlas para `chunks.embedding` (384 dimensiones, similitud coseno). Si Atlas M0 no permite crearlo desde el driver, el script imprime la definición JSON para crear el índice desde **Search & Vector Search** en Atlas. Atlas puede tardar un momento en dejar el índice listo.

## Ejecutar la API

```powershell
uvicorn app.main:app --app-dir backend --reload
```

- `GET /health`: comprueba la conexión a MongoDB (`503` si no está disponible; la API arranca de todos modos).
- `POST /search`: búsqueda textual en artículos con filtros opcionales de idioma y fechas.
- `POST /rag`: ruta reservada; responde `501` mientras se incorporan recuperación vectorial y proveedor LLM.
- `GET /docs`: documentación interactiva de FastAPI.

Ejemplo de `POST /search`:

```json
{
  "query": "sostenibilidad ambiental",
  "idioma": "es",
  "fecha_desde": "2024-01-01T00:00:00Z",
  "fecha_hasta": "2024-12-31T23:59:59Z",
  "limit": 10
}
```

## Notas de alcance

Esta entrega deja lista la infraestructura reproducible y el esqueleto de backend. Crear el cluster, usuario y regla de IP se hace en la cuenta Atlas del equipo. La ingesta de datos, generación de embeddings, ejecución de `$vectorSearch` desde la API y generación con LLM pertenecen a las siguientes etapas del proyecto.
