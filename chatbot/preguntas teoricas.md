# Diferencias entre 'completion' y 'chat' models
Un modelo completion está entrenando para seguir un hilo de pensamiento, mientras que un chat está entrenado para tener conversaciones.
Es la diferencia entre seguir un frase o párrafo (completion) y responder a una pregunta (chat)

# ¿Qué diferencias hay entre un modelo de razonamiento y un modelo generalista?
Un modelo de razonamiento planifica los pasos a seguir antes de ejecutarlos, un modelo generalista "simplemente" ejecuta. Los modelos de razonamiento son útiles cuando dar una respuesta requiera varios pasos (Por ejemplo, comparar precios de modelos de gpt). Sin embargo, para respuestas más sencilla no sólo son más lentos sino que pueden sobrepensar la respuesta y dar alucinaciones
# ¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?
Para ello se puede usar structured outputs. Usando una clase dict/json o Basemodel.
# RAG vs fine-tuning: ¿para qué sirve cada uno, y qué ventajas e inconvenientes tienen?
Un sistema RAG utiliza una base de datos para "actualizar" su conocimiento, mientras que finetuning require reentrenar la IA.

Los sistemas RAG son más útiles si la documentación se suele actualizar de forma más o menos frecuente. 

Una IA a la que se le ha hecho fine tuning actualiza su conocimiento interno. Esto hace que la IA rersponda más rápido y que el sistema sea más simple al tener menos componentes. Casos más útiles es cuando se utiliza un lenguaje técnico donde palabras comunes tienen otros significados. 
# ¿Qué es un agente?
Un agente es una IA generativa capaz de tomar acciones. La IA ya no simplemente da respuestas o ideas sino que puede además ejecutar sus acciones. Esto puede ser, acceder documentos, interactuar con internet, crear código, etc.
# ¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG? ¿Cómo evaluar el desempeño de una app de IA Generativa, en general: herramientas y métricas?
correctness
top k
