# Pasos necesarios a seguir.
Sería necesario recolectar los datos y etiquetarlos. Dependiendo de las cateogrías, se podría reaprovechar datos ya existentes con dataset publicos o re-aprovechando clases ya existentes (Si se quiere clasificar marcas de coches, se pueden hacer subset de la clase coche ya existente).

Para ahorrar en cantidad de datos a etiquetar se puede usar técnicas de data augmentation como rotaciones, añadir ruido, etc.

Será necesario reentrenar el modelo con los datos etiquetados.

# Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
Depediendo del caso de uso, puede surgir que las imágenes en producción disten de las de entrenamiento. Por ejemplo si se entrena con imágenes de internet de alta calidad pero la aplicación es de capturas con teléfonos móviles. Es importante tener claro cual es el caso de uso e intentar usar imágenes que se verán en producción.

Otro problema común es no tener suficientes ejemplos de las categorías. Esto puede dar casos en que objetos se categorizán mal. Para este problema es conveniente poder implementar el modelo en modo shadow para detectar problemas rápido. Cuando se detecten casos concretos, recolectar ejemplos del problema específico y reenrtenar.

# Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
La cantidad de datos está muy ligado a los tipos de categorías, número de categorías, tipo de imágenes, etc. Si se van a usar cámaras estáticas, no es útil tener imágenes con ángulos distintos, etc. Pero norma general, se puede llegar a necesitar varios cientos de imágenes por categoría.

Métricas útilos pueden ser la Intersección sobre Unión (IoU) dónde se evalúa el tener las cajas más pequeñas que contengan el objeto completo.

La detección de objectos suelen ser buenos en cuanto a detectar la localización aproximada de los objetos, pero es común que las cajas no sean perfectas.

# Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
- Data augmentation: Crear modificaciones en las imágenes de entrenamiento para generar nuevos datos viruatles. Esto incluye rotaciones, añadir ruido, cambios de color, deformación de la imagen etc.
- Uso de datos de la aplicación real: Si el modelo se va a usar en control de carreteras (Por ejemplo) usar imágenes capturadas por las cámaras disponibles. Así se reducirá la diferencia de precisión entre entrenamiento e inferencia.
- Reducción de categorías: Tener de forma determinada las categorías que son necesarias, reducimos las confusiones del modelo
- Entrenamiento de muestras enfrentadas: Un sistema de entrenamiento donde al modelo se le muestran dos (o tres) grupos de imágenes (Correcto/incorrecto ó Anclaje/correcto/incorrecto) y entrenar para vectorizar las imágenes similares de forma cercana y las distintas de forma separada