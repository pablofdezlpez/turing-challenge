# turing-challenge

|-chatbot
|   |-código ingesta
|   |-código chatbot
|   |-respuestas teóricas
|-object_detector
|   |-código object detector
|   |-dockerfile
|   |-futuras mejoras object detector
|- Pipfile
|_ Pipfile.lock

# chatbot
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
python chatbot/agent.py -m nombre-modelo -t temperatura
```
Por defecto modelo=gpt-5-nano y temperatura=0

consideraciones:
- Sólo admite modelos de OpenAI
- El agente es capaz de ejecutar código si se le pide

### Usando gracio

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
  