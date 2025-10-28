# Pasos necesarios a seguir.
Sería necesario recolectar los datos y etiquetarlos. Dependiendo de las cateogrías, se podría reaprovechar datos ya existentes con dataset publicos o re aprovechando clases ya existentes.
Para ahorrar en cantidad de datos a etiquetar se puede usar técnicas de data augmentation como rotaciones, añadir ruido, etc.

Será necesario reentrenar el modelo con los datos etiquetados.

# Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
Depediendo del caso de uso puede surgir que las imágenes en producción disten de las de entrenamiento. Por ejemplo si se entrena con imágenes de internet de alta calidad pero la aplicación es de capturas con teléfonos móviles.

Es importante tener claro cual es el caso de uso e intentar usar imágenes que se verán en producción.

Otro problema común es no tener suficientes ejemplos de las categorías pudiendo darse casos en que objetos se categorizán mal. Para este problema es conveniente poder implementar el modelo en modo shadow para detectar problemas rápido. Cuando se detecten casos concretos, recolectar ejemplos del problema específico.
# Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
La cantidad de datos está muy ligado a loss tipos de categorías, númeras de categorías, tipo de imágenes, etc.

Si se van a usar cámaras estáticas, no es útil tener imágenes con ángulos distintos, etc.

Por norma general, se puede llegar a neesitar varios cientos de imágenes por categoría.

Métricas útilos pueden ser la Intersección sobre Unión (IoU) dónde se evalúa el tener las cajas más pequeñas que contengan el objeto completo.

La detección de objectos suelen ser buenos en cuanto a detectar la localización aproximada de los objetos, pero es común que las cajas no sean perfectas.
# Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
