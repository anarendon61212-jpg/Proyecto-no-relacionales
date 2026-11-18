# Plan individual — Persona C: dataset y chunking

## Propósito

Preparar el contenido del dominio, implementar las dos estrategias de fragmentación y ejecutar la comparación controlada requerida por el proyecto. Este plan está ordenado por entregables y dependencias, con un plazo común de dos semanas para el equipo.

## Resultado esperado de tu rol

Entregar un dataset reproducible de al menos 100 artículos y 50 imágenes, módulos de chunking invocables desde el backend y evidencia comparativa que permita al equipo justificar qué estrategia funciona mejor para su dominio.

## Entregable C1 — Dataset curado

### Tareas

- [ ] Recolectar al menos 100 artículos relacionados con sostenibilidad, tecnología y energía renovable.
- [ ] Incluir contenido en español e inglés si las fuentes disponibles lo permiten; anotar idioma de forma consistente (`es`, `en`, etc.).
- [ ] Guardar título, contenido, idioma, fecha, categoría, fuente, URL/licencia cuando aplique e IDs o claves de imágenes asociadas.
- [ ] Recolectar al menos 50 imágenes pertinentes (p. ej. arquitectura sostenible, paneles, eólica, movilidad eléctrica).
- [ ] Conservar para cada imagen el archivo o URL accesible, descripción, categoría, fuente, licencia/atribución y relación con el artículo si existe.
- [ ] Evitar duplicados y contenido vacío; revisar fechas, idioma y metadatos requeridos por los validadores de MongoDB.
- [ ] Entregar JSON/manifiesto válido y un resumen de cantidades, fuentes y restricciones de uso.
- [ ] No incluir enlaces rotos ni imágenes que el equipo no tenga permiso de reutilizar en una demostración académica.

### Interfaz con A y B

- A congela nombres de campos, tipos y reglas. C debe validar contra ese contrato y notificar incompatibilidades antes de producir el lote final.
- B necesita una ruta estable del JSON/manifiesto y saber si las imágenes serán locales o URLs. Acuerden una ubicación común y no guarden rutas absolutas del computador de una sola persona.

## Entregable C2 — Estrategia sentence-aware

### Parámetros de partida

- Máximo de 5 oraciones por chunk.
- Solapamiento de 1 oración entre chunks consecutivos.
- Conservar el orden y evitar chunks vacíos. Decidir y documentar cómo tratar títulos, listas, saltos y textos sin puntuación fiable.

### Tareas

- [ ] Implementar una función que reciba texto y devuelva una lista de chunks en orden.
- [ ] Comprobar manualmente artículos cortos, largos, con abreviaturas y puntuación de ambos idiomas.
- [ ] Guardar parámetros y versión de implementación en documentación o configuración.

## Entregable C3 — Estrategia semantic

### Parámetros de partida

- Empezar con el umbral de similitud sugerido de 0.80.
- Agrupar oraciones consecutivas por embeddings de oración; documentar si el corte ocurre cuando la similitud coseno cae por debajo del umbral.
- Definir un tamaño máximo y mínimo operativo para evitar un único chunk gigante o fragmentos demasiado pequeños. Mantener esos límites iguales al comparar, salvo que se justifique lo contrario.
- Usar el modelo/text-embedding acordado con B para evitar dimensiones o espacios vectoriales diferentes entre consulta y documento.

### Tareas

- [ ] Implementar función separada sentence-aware/semantic detrás de una interfaz común.
- [ ] Probar un conjunto de ejemplos manuales para revisar que los cambios temáticos generen cortes razonables.
- [ ] Registrar versión, umbral, modelo auxiliar y límites efectivos.
- [ ] Entregar resultados en el formato pactado con B: lista ordenada de textos más metadatos necesarios; B se encarga de persistir y asociar embeddings.

## Entregable C4 — Diez consultas experimentales

### Tareas

- [ ] Preparar diez preguntas reproducibles sobre el corpus antes de mirar qué estrategia gana.
- [ ] Incluir búsquedas semánticas, filtros por idioma/fecha y preguntas que requieran sintetizar más de un artículo.
- [ ] Cubrir las preguntas obligatorias del enunciado dentro de las pruebas y demostración.
- [ ] Entregar la consulta, intención, filtros esperados y artículos/chunks relevantes conocidos si se puede identificar una referencia manual.

### Semilla de consultas

Adapta estas preguntas al contenido real curado; deben tener respuesta/evidencia suficiente en el dataset:

1. ¿Qué documentos hablan de sostenibilidad ambiental?
2. ¿Qué artículos en inglés sobre tecnología se publicaron en 2024?
3. ¿Cuáles son las principales tendencias de energía renovable descritas en los documentos?
4. ¿Qué ventajas y límites de la energía solar aparecen en las fuentes?
5. ¿Cómo comparan los artículos energía solar y eólica?
6. ¿Qué tecnologías se mencionan para mejorar la eficiencia energética?
7. ¿Qué relación describen las fuentes entre movilidad eléctrica y sostenibilidad?
8. ¿Qué soluciones aparecen para reducir emisiones en ciudades?
9. ¿Qué ejemplos de arquitectura sostenible se describen o ilustran?
10. ¿Qué políticas o iniciativas climáticas se mencionan y qué objetivo tienen?

La pregunta de imágenes similares se usa además en la demostración visual; no se mezcla con la evaluación de estrategias de chunking textual.

## Entregable C5 — Comparación y resultados

### Tareas

- [ ] Ejecutar las mismas diez preguntas contra los chunks sentence-aware y semantic.
- [ ] No variar consulta, filtros, límite de resultados ni modelo durante la comparación.
- [ ] Guardar los chunks/documentos recuperados por pregunta y estrategia.
- [ ] Evaluar calidad con una rúbrica acordada con A: relevancia del contexto recuperado, cobertura de la respuesta y utilidad/precisión percibida.
- [ ] Contar chunks por estrategia y calcular longitud promedio (indicar si la longitud es palabras, tokens o caracteres).
- [ ] Registrar casos sin evidencia y errores; no presentar una respuesta generada como correcta si no coincide con las fuentes.
- [ ] Redactar una conclusión argumentada, incluyendo límites del experimento y qué estrategia recomendarías para este corpus.

### Producto

CSV/JSON de resultados, tabla comparativa y párrafo de interpretación para el informe de A.

## Criterios para dar por terminado tu aporte

- Dataset de 100+ artículos y 50+ imágenes, con fuentes y atribuciones.
- Cada artículo satisface los campos obligatorios del contrato.
- Dos chunkers ejecutables con parámetros registrados y resultados no vacíos.
- Cada chunk conserva su estrategia para compararlo dentro de MongoDB.
- Diez preguntas fijas ejecutadas bajo condiciones comparables para ambas estrategias.
- Tabla con cantidad de chunks, longitud promedio, resultados y calidad percibida.
- Fuentes/archivos entregados en rutas reproducibles y no dependientes de tu computador.

## Riesgos y coordinación

- No cambiar el esquema después de que B implemente la ingesta sin avisar a A y B.
- No generar embeddings de texto con un modelo en una persona y las consultas con otro: recuperación vectorial requiere que consulta y corpus estén en el mismo espacio vectorial.
- Los embeddings de imagen deben generarse con un modelo visual/multimodal acordado con B; las dimensiones se documentan aparte de las de texto.
- El umbral semantic 0.80 es un parámetro inicial del enunciado, no una garantía de mejor calidad; informar los resultados observados.
- Las URLs de imágenes pueden fallar o requerir internet; conservar un plan de demostración que funcione con el material disponible.
