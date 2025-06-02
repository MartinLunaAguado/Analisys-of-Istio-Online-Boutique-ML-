## 1. Preparación y limpieza de datos

Se cargan los datos originales de Istio desde un CSV.
Se rellenan valores nulos, se normalizan columnas y se crea una columna 'result' que clasifica cada petición como 'success' o 'error' según varios criterios.
Los datos se ordenan por origen, destino y timestamp, y se guardan en un nuevo archivo.

## 2. Cálculo de métricas diferenciales y separación de éxitos/errores

Se cargan los datos agregados y se convierten los timestamps.
Se separan los éxitos y errores (HTTP y gRPC), calculando para cada grupo la diferencia de peticiones, bytes y duración respecto al registro anterior.
Se calcula la latencia y se guardan los resultados de éxito y error en archivos separados.
Finalmente, se fusionan todos los datos y se guardan en un archivo final.

## 3. Agregación de métricas por timestamp

Se agrupan los datos por origen, destino y timestamp.
Se calculan tasas de éxito/error, recuentos, duraciones y latencias medias, así como bytes transferidos.
El resultado se guarda en un archivo CSV.

## 4. Resampleo y cálculo de KPIs en ventanas temporales

Se definen varias ventanas de tiempo (15s, 30s, 1min, 5min, 10min).
Para cada ventana, se agrupan y agregan métricas clave (throughput, request_rate, etc.).
Se eliminan registros de un punto de partida específico y se guarda el resultado.

## 5. Cálculo de percentiles de latencia

Se reagrupan los datos en intervalos mayores y se calculan percentiles de latencia (p50, p90, p95, p99) ponderados por el número de peticiones.
El resultado se guarda en un archivo CSV.

## 6. Clasificación de enlaces y nodos según anomalías

Se añade una columna de color de enlace y una clase de anormalidad basada en el success_rate.
Se guarda el DataFrame procesado.

## 7. Visualización de la evolución del grafo con NetworkX

Se crea una animación de la evolución del grafo de microservicios a lo largo del tiempo, coloreando los enlaces según anomalías.
Se guarda como GIF.

## 8. Visualización de snapshots de grafos por minuto

Se generan y visualizan grafos de conocimiento para intervalos de 1 minuto, mostrando si hay anomalías en cada ventana.

## 9. Construcción y análisis de grafos con Raphtory

Se crea un grafo temporal con Raphtory, añadiendo nodos y aristas con todas las propiedades relevantes.
El grafo se guarda en disco.

## 10. Consultas GraphQL sobre el grafo

Se lanza un servidor GraphQL para consultar el grafo.
Se realizan queries para obtener propiedades de nodos, aristas, nodos de alto grado y el esquema del grafo.

## 11. Análisis temporal y algoritmos de grafos

Se realiza un análisis rolling window sobre el grafo con Raphtory y NetworkX.
Se calcula el PageRank y las comunidades en cada ventana temporal.
Los resultados se guardan en archivos CSV.

## 12. Visualización de métricas de rendimiento

Se generan gráficas de métricas (anomalías, success_rate, throughput, request_rate, latencia) para cada relación origen-destino.
Se crean animaciones GIF de la evolución temporal de estas métricas.

## 13. Detección de relaciones con anomalías

Se identifican y muestran las relaciones origen-destino con anomalías detectadas.

## 14. Visualización de PageRank y comunidades

Se grafican la evolución temporal del PageRank de los nodos más importantes y el número/tamaño de comunidades detectadas.

## 15. Lanzamiento del servidor Raphtory

Se inicia el servidor GraphQL de Raphtory para explorar el grafo desde el navegador.
Resumen:
El notebook work1.ipynb implementa un pipeline completo de procesamiento, análisis y visualización de datos de microservicios, desde la limpieza y agregación de datos crudos hasta la construcción de grafos temporales, análisis de anomalías, cálculo de métricas avanzadas y visualización interactiva y animada de los resultados, utilizando tanto NetworkX como Raphtory. Además, permite explorar el grafo mediante queries GraphQL y analizar la evolución de la red y sus métricas clave a lo largo del tiempo.