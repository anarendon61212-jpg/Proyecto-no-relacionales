# Plan individual — Persona A: análisis y modelado de datos

## Propósito

Entregar la definición funcional y el diseño NoSQL del Sistema RAG, y coordinar la documentación final para que el modelo de datos coincida con el dataset, el backend, la búsqueda de imágenes y el experimento de chunking.

Este documento asigna responsabilidades y productos de trabajo para el proyecto final en un plazo total de dos semanas. Está organizado por entregables, no por calendario. La Persona A no debe inventar otra arquitectura en paralelo: los nombres y tipos acordados en el contrato de datos son los que implementarán B y C.

## Resultado esperado de tu rol

Al terminar, el equipo debe tener un esquema justificado y validable, un análisis del sistema y un informe final unido con evidencia de la implementación y evaluación.

## Entregable A1 — Análisis y requisitos

### Tareas

- [ ] Redactar el universo del discurso: qué información trata el sistema, quién lo usará y qué consultas resuelve.
- [ ] Definir el alcance: artículos sobre tecnología, sostenibilidad y energías renovables, con imágenes relacionadas.
- [ ] Convertir los requisitos del enunciado en una lista verificable: ingesta, búsquedas, RAG, imágenes, chunking, datos mínimos, API y demostración.
- [ ] Separar requisitos obligatorios de componentes opcionales, como historial de usuarios o RAGAS.
- [ ] Definir las cinco consultas demostrativas finales, incluyendo búsqueda semántica, filtros por metadatos, pregunta RAG, similitud de imágenes y un caso multimodal.

### Producto

Sección de análisis del informe final con alcance, actores, consultas y criterios de aceptación.

## Entregable A2 — Contrato de datos y modelo NoSQL

### Tareas

- [ ] Acordar con B y C los nombres de colecciones y campos antes de que se consoliden el dataset o la ingesta.
- [ ] Documentar un ejemplo JSON válido para cada colección acordada.
- [ ] Especificar campos requeridos, tipos, formatos de fecha, valores permitidos y referencias entre documentos.
- [ ] Justificar qué datos se embeben y cuáles se referencian: por ejemplo, chunks en colección independiente, referencias entre artículos e imágenes, y metadatos pequeños embebidos.
- [ ] Explicar por qué no se almacenan archivos de imagen grandes dentro de los documentos; acordar si se guardará una URL, ruta o ubicación externa.
- [ ] Revisar que el esquema acepte `estrategia_chunking`, `embedding`, `modelo`, fecha de ingesta y enlace al artículo padre.
- [ ] Acordar campos de imagen para búsqueda visual y multimodal. No asumir que los embeddings de texto y de imagen son intercambiables: el equipo debe definir el modelo y las dimensiones antes de crear los índices.
- [ ] Revisar los validadores existentes del inicializador y reportar a B cualquier diferencia antes de que el dataset sea cargado.

### Contrato inicial sugerido

- `articulos`: `titulo`, `contenido_texto`, `idioma`, `fecha`, `categoria`, `fuente`, `imagenes_relacionadas`.
- `chunks`: `doc_id`, `chunk_index`, `estrategia_chunking`, `chunk_texto`, `embedding`, `modelo`, `fecha_ingesta`.
- `imagenes`: `url` o ubicación del archivo, `descripcion`, `categoria`, `fuente` y campos de embedding/modelo definidos con B.
- `usuarios`: `usuario_id` e `historial` embebido solo si el equipo confirma que el tiempo alcanza y el caso se usará en la demostración.

Los nombres anteriores son un punto de partida; cualquier cambio se comunica en un único contrato compartido y se congela antes de la ingesta final.

### Producto

Una sección de diseño con diagramas o tabla de colecciones, ejemplos JSON, reglas de validación y justificación de embedded, referenced e híbrido.

## Entregable A3 — Diseño de consultas e índices

### Tareas

- [ ] Mapear cada consulta de la aplicación a su fuente y mecanismo: texto, vector, filtros, agregación o recuperación de imágenes.
- [ ] Revisar con B que las consultas importantes tengan índices correspondientes y que los campos de prefiltrado vectorial estén incluidos en el índice de Vector Search.
- [ ] Documentar como mínimo el compuesto fecha/idioma, el índice de texto y los índices vectoriales que finalmente use el proyecto.
- [ ] No presentar un índice como creado hasta comprobar su estado en Atlas/Compass; distinguir entre planificado y listo.

## Entregable A4 — Informe final integrado

### Tareas

- [ ] Ensamblar el análisis de requisitos, diseño NoSQL, infraestructura, instalación y manual de ejecución.
- [ ] Incorporar de C el tamaño y procedencia del dataset, las diez preguntas y la tabla comparativa de chunking.
- [ ] Incorporar de B la arquitectura, endpoints, instrucciones de configuración y ejemplos de ejecución.
- [ ] Explicar los resultados y limitaciones observadas; no afirmar que el modelo respondió con evidencia si no se guardaron las fuentes recuperadas.
- [ ] Incluir la comparación con enfoque relacional solicitada por el proyecto.
- [ ] Revisar consistencia entre nombres/campos del informe, JSON, validadores, código y capturas.

### Producto

Informe final listo para entregar y revisión de que el README permite a otra persona levantar y probar el proyecto.

## Coordinación y dependencias

1. **Primero:** acordar campos y colecciones con B y C. C necesita el contrato para generar los archivos; B necesita el mismo contrato para validar e ingerir.
2. **Después:** recibir de C los datos y resultados experimentales; recibir de B evidencia de conexión, índices y endpoints.
3. **Al cierre:** integrar todo, señalar limitaciones reales y comprobar que cada requisito del enunciado tiene evidencia.

## Criterios para dar por terminado tu aporte

- El informe explica para quién es el sistema y qué problema resuelve.
- Cada colección tiene un ejemplo consistente con validadores y datos reales.
- Las decisiones de modelado tienen justificación ligada a las consultas.
- El informe separa claramente lo implementado de lo pendiente o limitado.
- Los resultados de chunking provienen de ejecutar el mismo conjunto de preguntas con ambas estrategias.
- La comparación relacional, la demostración y las instrucciones de ejecución están incluidas.

## Riesgos que debes vigilar

- El esquema de la Persona A y el formato JSON de C divergen.
- Se documenta búsqueda multimodal sin un modelo/índice visual realmente configurado.
- Se redactan conclusiones antes de que B y C entreguen resultados medidos.
- Se deja el informe para el final y no se reserva tiempo de integración.

## Fuentes técnicas de referencia

- Enunciado base del equipo: `Proyecto_Sistema RAG NoSQL con MongoDB_2026.pdf`.
- [MongoDB Vector Search: filtros y consultas](https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/).
- [MongoDB Schema Validation](https://www.mongodb.com/docs/manual/core/schema-validation/).
