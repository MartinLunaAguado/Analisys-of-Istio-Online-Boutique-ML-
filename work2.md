# Explicación detallada del código en work2.ipynb

Este notebook implementa un flujo completo para la construcción, análisis y visualización de grafos de microservicios a partir de métricas extraídas de Istio/Kiali, utilizando tanto la librería Raphtory como Pyvis. A continuación se explica cada bloque relevante:

## 1. Inicialización del servidor Raphtory
Se lanza un servidor GraphQL de Raphtory para poder consultar y explorar grafos temporalmente enriquecidos desde una interfaz web. Esto permite analizar la evolución de los microservicios y sus relaciones a lo largo del tiempo.

## 2. Creación del grafo de "creación de microservicios"
Se carga un CSV con información de microservicios y se construye un grafo donde:
- Cada microservicio es un nodo, con propiedades como la clase de anormalidad y color visual.
- Cada pod es también un nodo.
- Se añaden aristas que representan la relación de despliegue entre microservicio y pod.
- El grafo se guarda en disco para su posterior análisis temporal.

## 3. Visualización de columnas del dataset
Se imprime la lista de columnas del archivo de métricas procesadas para verificar la estructura de los datos antes de construir los grafos.

## 4. Instalación de dependencias
Se instalan las librerías necesarias, como Pyvis para la visualización interactiva de grafos en HTML.

## 5. Construcción de grafos enriquecidos con Raphtory
Se crean dos variantes de grafos temporales:
- Uno con nodos y aristas que incluyen propiedades métricas (latencia, tasas de error, etc.) y etiquetas visuales según la clase de anormalidad.
- Otro que además añade propiedades temporales a las aristas, permitiendo consultas sobre la evolución de las métricas entre pares de microservicios.

## 6. Consultas temporales sobre el grafo
Se cargan los grafos guardados y se realizan consultas como:
- Grado de un nodo (cuántos microservicios conecta).
- Grado antes de un instante dado.
- Grado en una ventana temporal.
- Evolución temporal de la latencia entre dos microservicios.

## 7. Visualización interactiva con Pyvis
Se utiliza Pyvis para crear un grafo HTML interactivo donde:
- Cada nodo representa un microservicio.
- El color de los nodos y enlaces depende de la clase de anormalidad.
- Se asegura que solo haya un enlace por cada par único de microservicios.
- Se calcula el porcentaje de peticiones normales para cada par y, si es menor al 95%, el enlace se dibuja en rojo; si es igual o mayor, en verde.
- El grafo se exporta a un archivo HTML para su visualización en navegador.

## 8. Visualización de solo anomalías
Se genera un grafo alternativo mostrando únicamente los nodos y enlaces anómalos, facilitando la identificación de problemas en la arquitectura de microservicios.

## 9. Instrucciones para visualizar el grafo
Se incluyen instrucciones para lanzar un servidor HTTP local y visualizar el grafo HTML generado desde el navegador.

---

**Resumen:**
El notebook permite desde la carga y procesamiento de métricas, la construcción de grafos temporales enriquecidos, la consulta temporal avanzada y la visualización interactiva, facilitando el análisis de la salud y relaciones de los microservicios en un entorno de Kubernetes/Service Mesh.
