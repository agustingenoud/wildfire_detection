# Wildfire detection

The dataset with Yolo form should be placed inside the datasets dir.
The data.yaml file should be placed in the root.

```
|___/datasets
|       |_____/test
|       |       |_____/images
|       |       |_____/labels
|       |
|       |_____/train
|       |       |_____/images
|       |       |_____/labels
|       |
|       |_____/valid
|                |_____/images
|                |_____/labels
|
|___data.yaml
````

Los primeros modelos train 1-6 fueron para establecer un baseline y entender cómo Yolo se comporta. Luego realicé los siguientes cambios en los hyperpsrámetros para poder obtener mejores resultados. Los siguientes hp's fueron utilizados logrando una mejor convergencia en mayor cantidad de corridas (50):

        epochs=50
        batch=64
        dropout=0.25
        optimizer="AdamW"
        lr0=1e-3
        lrf=1e-2

Llegando a obtener un resultado sobre el dataset de validación de 0.935 mAP.

## Follow Up Questions
- ¿Cómo podrías hacer para mejorar la performance del modelo? ¿Que ideas se te ocurren?
    - El dataset no posee imágenes que no tengan humo, podría elaborar estrategias específicas para poder mejorar los falsos positivos.


- Si tuvieras que detectar nubes además de columnas de humo, ¿Cómo trabajarías con eso?
  - Empezaría por probar algoritmos ya utilizados para imágenes satelitales (Fmask, Sen2Cor, MAJA), si bien los dispositivos de captura son distintos a cámaras de viiglancia podrían ser un buen punto de partida para obtener un baseline.