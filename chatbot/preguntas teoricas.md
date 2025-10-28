# Diferencias entre 'completion' y 'chat' models
Un modelo completion está entrenando para seguir un hilo de pensamiento, mientras que un chat está entrenado para tener conversaciones.
Es la diferencia es equivalente entre seguir un frase o párrafo (completion) y responder a una pregunta (chat).

# ¿Qué diferencias hay entre un modelo de razonamiento y un modelo generalista?
Un modelo de razonamiento planifica los pasos a seguir antes de ejecutarlos, un modelo generalista "simplemente" ejecuta. Los modelos de razonamiento son útiles cuando dar una respuesta requiera varios pasos (Por ejemplo, comparar precios de productos en internet). Sin embargo, para respuestas más sencillas no sólo son más lentos sino que pueden sobrepensar la respuesta y dar lugar a alucinaciones.

# ¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?
Para ello se puede usar structured outputs. Usando una clase dict/json o Basemodel. 

# RAG vs fine-tuning: ¿para qué sirve cada uno, y qué ventajas e inconvenientes tienen?
Un sistema RAG utiliza una base de datos para "actualizar" su conocimiento, mientras que finetuning reentrena la IA para actualizar su conocimiento o lenguaje.

Los sistemas RAG son más útiles si la documentación se suele actualizar de forma más o menos frecuente y no se requiere salto logicos. Son especialemnte útiles cuando se require información corroborada de una base de datos grandes. Esto ayuda a verificar alucinaciones al igual que ahorra tiempo de entrenamiento. 

Una IA a la que se le ha hecho fine tuning actualiza su conocimiento interno. Esto hace que la IA rersponda más rápido y que el sistema sea más simple al tener menos componentes. Casos más útiles es cuando se utiliza un lenguaje técnico donde palabras comunes tienen otros significados o si debe aprender nuevos razonamientos. Esto sería el caso de una IA especializada en ingerniería o en un idioma no soportado pro defecto.

# ¿Qué es un agente?
Un agente es una IA generativa capaz de tomar acciones. La IA ya no simplemente da respuestas o ideas sino que puede además ejecutar sus acciones. Esto puede ser, acceder documentos, interactuar con internet, crear código, etc. Para ello se usa una IA tipo chat y se le da a acesso a herramientas usando tools.

# ¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG? ¿Cómo evaluar el desempeño de una app de IA Generativa, en general: herramientas y métricas?
Para poder evaluar un agente es imprescindible tener un trackeo de los hilos de conversación, para ello es útil tener telemetría desde la entrada hasta la respues final del modelo. Para ello se pueden usar libreríua como langsmith o langfuse (Y más que van saliendo cada mes)

Para evluar un modelo Q&A se puede usar:
- Precisión: Cuanto de la respuesta del modelo existe realmente en el texto extraido. Esto es especialmente importante en modelos que deben respaldar sus respuestas.
- BERTScore: Igualdad semántica entre las respuesta del modelo y la información del texto. Esta métrica ayuda a que el modelo puede parafrasear el texto.
- Se pueden plantear métricas cualitativas como tono, veracidad o relevancia. Para estas métricas es necesario etiquetado manual o, si el coste no es un limitante, tener una IA juez que evalua la respuesta según el contexto
  
Para el sistema de retrieval:
- Recall@k: % de veces que el documento que se buscaba estaba entre los k resultados
- Precision@k: De los k documentos obtenidos, cuantos son relevantes

Evaluar un sistema de AI generativa completo tendrá en cuenta todas estas métricas y usar las herramientas mencionadas. Además se pueden plantear métricas pseudo cualitativas como n de veces que el usuario ha debido preguntar hasta obtener la respuesta correcta.
