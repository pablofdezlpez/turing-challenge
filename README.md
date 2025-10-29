# turing-challenge
```
|-chatbot
|   |-config.yaml <-- archivo configuración para ingesta y chatbot
|   |-agent.py <- código del chatbot incluyendo nodos y ejes
|   |-ingest.py <- código de la ingesta
|   |-prompts.py <- colección de prompts usados tanto en ingesta como chatbot
|   |-tools.py <- definición de tools usadas por el agente
|   |-user_interface.py <- código de gradio
|   |-respuestas teóricas
|-object_detector
|   |-app.py <-código fastapi
|   |-main.py <-código fuente object detector
|   |-dockerfile
|   |-futuras mejoras object detector
|- Pipfile
|_ Pipfile.lock
```
# chatbot
Usar config file para definir modelo, localización del vector store, número de documentos a retirar, etc.
## Como hacer ingesta? 

Para realizar un trabajo de ingesta
```
python chatbot/ingest.py --d directory/of/files
```
Consideraciones:
- Saltará un error si alguno de los documentos no está en formato .pdf
- Cuando se detecte un archivo con cv en su nombre se extraerá información básica de forma estructurada y se imprimirá en consola, pero esta no se almacenará
- Las imágenes de los archivos serán descritas por IA y alamcenadas 
  
## Uso de chatbot
### Usando consola
```
python chatbot/agent.py 
```
consideraciones:
- El agente es capaz de ejecutar código si se le pide

### Usando gradio

```
python chatbot/user_interface.py
```

# Object detection
## Construir imagen
```
docker build -t tag-name .
docker run -p 8000:8000 tag-name
curl -X POST "http://localhost:8000/infer" -F "file=@filename" --output outputname
```
consideraciones:
- file debe apuntar a una imagen de cualquier formato
- resultado será un json con los objetos detectados y sus clases
  
