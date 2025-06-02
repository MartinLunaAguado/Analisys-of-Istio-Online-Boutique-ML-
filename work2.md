# Explicación detallada de work2.ipynb

Este notebook implementa un pipeline de análisis y visualización de datos de microservicios, centrado en la construcción y exploración de grafos temporales usando Raphtory y PyVis. A continuación se describe el flujo y propósito de cada bloque:

## 1. Inicio del servidor Raphtory
Se lanza un servidor GraphQL de Raphtory para explorar grafos guardados en disco. Esto permite realizar queries y análisis interactivos sobre los grafos generados.

## 2. Creación de un grafo de creación de microservicios
Se carga un dataset CSV con información de microservicios y pods. Para cada microservicio, se identifica el primer instante en que aparece y se crea un nodo con propiedades de anomalía (color y etiqueta). Se conecta cada microservicio a su pod mediante una arista. El grafo se guarda en disco para su posterior análisis.

## 3. Visualización de columnas del dataset
Se muestra el listado de columnas de un CSV procesado para facilitar la exploración y selección de variables relevantes.

## 4. Instalación de PyVis
Se instala la librería PyVis para la visualización interactiva de grafos en HTML.

## 5. Construcción de un grafo enriquecido de microservicios
Se crea un grafo temporal donde cada nodo representa un microservicio y cada arista una comunicación. Se añaden propiedades como tasas de éxito/error, throughput, latencia, etc. Los nodos y aristas se etiquetan y colorean según la presencia de anomalías. El grafo se guarda para análisis temporal.

## 6. Variante con actualización de propiedades temporales
Se implementa una versión donde las aristas se actualizan con propiedades temporales usando `add_updates`, permitiendo análisis históricos más precisos.

## 7. Consultas temporales sobre el grafo
Se cargan grafos desde disco y se realizan consultas sobre la conectividad de un nodo específico (por ejemplo, `checkoutservice`) en distintos intervalos temporales. También se analiza la evolución de la latencia entre dos servicios a lo largo del tiempo.

## 8. Cálculo de latencia media en ventanas móviles
Se calcula y muestra la latencia media entre dos servicios usando ventanas móviles (rolling window), lo que permite observar la evolución temporal de la métrica.

## 9. Visualización de la evolución de la latencia
Se grafica la evolución de la latencia media entre dos servicios usando Matplotlib, mostrando cómo varía en ventanas de 3 minutos.

## 10. Creación de variantes de grafos enriquecidos
Se generan variantes del grafo enriquecido, cambiando el nombre y ruta de guardado, para comparar diferentes configuraciones o datasets.

## 11. Instalación de Raphtory con visualización
Se instalan dependencias adicionales para habilitar la visualización avanzada con Raphtory.

## 12. Visualización interactiva con PyVis
Se utiliza PyVis para crear una visualización HTML interactiva del grafo de microservicios, coloreando nodos y aristas según anomalías y mostrando información relevante al pasar el ratón.

## 13. Visualización agregada y por anomalías
Se crea una visualización agregada donde el color de la arista depende de la frecuencia de anomalías. También se genera una visualización solo con nodos y enlaces anómalos.

## 14. Instrucciones para servir el HTML
Se dan instrucciones para abrir una terminal y servir el archivo HTML generado con un servidor HTTP local.

---

**Resumen:**
El notebook work2.ipynb permite construir, analizar y visualizar grafos temporales de microservicios, identificando anomalías y explorando la evolución de métricas clave. Utiliza Raphtory para el modelado temporal y PyVis para la visualización interactiva, facilitando tanto el análisis automático como la exploración manual de los datos y grafos generados.