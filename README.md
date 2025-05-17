# Análisis y Procesamiento de Métricas de Istio

Este proyecto contiene un flujo completo de procesamiento, análisis y visualización de métricas de tráfico de red en un entorno de microservicios gestionado por Istio. El flujo está implementado en el archivo `work1.ipynb` y abarca desde la limpieza de datos hasta la visualización avanzada de grafos y métricas.

## Flujo General

1. **Carga y Limpieza de Datos**
   - Se carga el archivo CSV original (`data/istio_request_2.2.csv`).
   - Se rellenan valores nulos y se normalizan columnas relevantes.
   - Se clasifica cada fila como 'success' o 'error' según los códigos de respuesta y flags.
   - Se ordenan los datos y se guarda un archivo intermedio (`aggregated_istio_data.csv`).

2. **Cálculo de Métricas Diferenciales**
   - Se calculan diferencias entre registros consecutivos para obtener nuevas métricas por ventana temporal (peticiones, bytes, duración).
   - Se separan los casos de éxito y error, generando archivos específicos para cada tipo de error.
   - Se combinan todos los resultados en un archivo final (`new_request_istio_data.csv`).

3. **Agregación de Métricas**
   - Se agrupan los datos por origen, destino y timestamp.
   - Se calculan tasas de éxito/error, recuentos, duraciones y bytes transferidos.
   - El resultado se guarda en `aggregated_istio_rates.csv`.

4. **Cálculo de KPIs y Ventanas Temporales**
   - Se generan KPIs (throughput, request rate, etc.) para diferentes ventanas temporales (15s, 30s, 1min, 5min, 10min).
   - Se eliminan registros no válidos y se guarda el resultado en `kiali_kpi_metrics.csv`.

5. **Cálculo de Percentiles de Latencia**
   - Se agrupan los datos en ventanas mayores y se calculan percentiles de latencia (p50, p90, p95, p99).
   - El resultado se almacena en `kiali_latency_percentiles.csv`.

6. **Clasificación de Anomalías**
   - Se añade una columna de color y clase de anormalidad según la tasa de éxito.
   - El archivo procesado se guarda como `kiali_kpi_metrics_processed.csv`.

7. **Visualización de Grafos de Comunicación**
   - Se generan animaciones y visualizaciones de la evolución del grafo de servicios usando NetworkX y Matplotlib.
   - Se identifican relaciones anómalas y normales en el tráfico.

8. **Construcción y Consulta de Grafos con Raphtory**
   - Se construye un grafo temporal con Raphtory a partir de los datos procesados.
   - Se ejecutan queries GraphQL para explorar propiedades de nodos y aristas, así como detectar nodos de alto grado y realizar introspección del esquema.

9. **Visualización de Métricas**
   - Se generan gráficos de evolución temporal para cada relación origen-destino, mostrando anomalías, tasas de éxito, throughput, request rate y latencia.

10. **Validación de Anomalías**
    - Se valida la coherencia de las anomalías detectadas por minuto y par origen-destino.

## Archivos Generados
- `aggregated_istio_data.csv`: Datos limpios y clasificados.
- `new_request_istio_data.csv`: Métricas diferenciales por evento.
- `aggregated_istio_rates.csv`: Métricas agregadas por ventana temporal.
- `kiali_kpi_metrics.csv`: KPIs por ventana temporal.
- `kiali_latency_percentiles.csv`: Percentiles de latencia.
- `kiali_kpi_metrics_processed.csv`: Datos con clasificación de anomalías.
- `evolucion_grafo.gif`: Animación de la evolución del grafo de servicios.
- `graphs/kiali_fullnode_graph/`: Grafo temporal para consultas Raphtory.

## Requisitos
- Python 3.x
- Pandas, NumPy, Matplotlib, NetworkX, Raphtory

## Ejecución
1. Instala las dependencias necesarias:
   ```bash
   pip install pandas numpy matplotlib networkx raphtory
   ```
2. Ejecuta las celdas del notebook `work1.ipynb` en orden.
3. Explora los archivos generados y las visualizaciones para analizar el comportamiento de la red de microservicios.

## Notas
- El flujo está diseñado para ser reproducible y modular, permitiendo adaptar cada etapa a nuevos datasets o métricas.
- Las visualizaciones y análisis permiten identificar cuellos de botella, anomalías y patrones de tráfico en arquitecturas de microservicios gestionadas por Istio.
