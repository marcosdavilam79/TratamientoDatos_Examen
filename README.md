<h1 style="color: #FF0000;">MATRIZ DE CONFUSIÓN - PROYECTO</h1>


# Importación de bibliotecas necesarias
   ```python
    import torch                                       # Para trabajar con redes neuronales
    import torch.nn as nn                              # Módulo de redes neuronales de PyTorch
    from torchvision import transforms, datasets       # Para transformaciones de datos y conjuntos de datos
    import numpy as np                                 # Para manipulación de matrices y arreglos
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sns                              # Biblioteca para graficar
    import matplotlib.pyplot as plt
    from torchvision import models                     # Importa modelos preentrenados desde torchvision****
```
# Define las rutas a las carpetas de datos de entrenamiento y prueba
```python
    train_data_dir = './train'
    test_data_dir = './test'
```
# Define las transformaciones de aumento de datos para el conjunto de entrenamiento
  ```python
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),                   # Volteo horizontal aleatorio
        transforms.RandomVerticalFlip(),                     # Volteo vertical aleatorio
        transforms.RandomRotation(30),                       # Rotación aleatoria en un rango de ±30 grados
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Recorte aleatorio y redimensionamiento
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Afinamiento aleatorio
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Ajustes de color
        transforms.RandomGrayscale(p=0.2),                  # Transformación a escala de grises aleatoria
        transforms.ToTensor(),                              # Convierte la imagen en un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza los valores de píxeles
    ])
```
# Carga los datos de entrenamiento y prueba para imágenes en color
  ```python
# Crea conjuntos de datos de entrenamiento y prueba utilizando la clase ImageFolder de torchvision, la cual organiza los datos en subdirectorios donde cada uno de ellos corresponde a una clase.
    train_dataset_color = datasets.ImageFolder(train_data_dir, transform=data_transforms)
    test_dataset_color = datasets.ImageFolder(test_data_dir, transform=data_transforms)
```
# Imprime la cantidad de imágenes encontradas y con las que se trabajará
  ```python
    print(f'Se encontraron {len(train_dataset_color)} imágenes para entrenamiento pertenecientes a {len(train_dataset_color.classes)} clases.')
    print(f'Se encontraron {len(test_dataset_color)} imágenes para entrenamiento pertenecientes a {len(test_dataset_color.classes)} clases.')
```
# Cargar el modelo preentrenado MobileNet
  ```python
    model_color = models.mobilenet_v2(pretrained=True)         #Modelo preentrenado diseñado para la clasificación de imágenes.
```
# Congelar todas las capas del modelo MobileNet
  ```python
# Se realiza un bucle a través de todos los parametros, congela o retiene los datos de todas las capas excepto de la final, la cual se encarga de la clasificación de nuestro conjunto de datos.
    for param in model_color.parameters():
        param.requires_grad = False
```
# Modificar la capa final para ajustar al número de clases en el conjunto de datos
  ```python
#Reemplaza la capa final de la red preentrenada por la nuestra y así se ajusta a nuestro numero de clases en este caso a 8.
    num_classes = len(train_dataset_color.classes)
    model_color.classifier[1] = nn.Linear(model_color.classifier[1].in_features, num_classes)
```
# Define la función de pérdida y el optimizador para el modelo en color
  ```python
    criterion = nn.CrossEntropyLoss()                         #Se encarga de calcular la pérdida de datos con el objetivo de minimizar esto y que la predicción sea más precisa.
    optimizer_color = torch.optim.Adam(model_color.classifier[1].parameters(), lr=0.001)     #Optimiza los pesos del modelo durante el entrenamiento, lr=0.001 especifica la tasa de aprendizaje, mientras mas alta sea el entrenamiento es menos preciso pero acelera el entrenamiento y si es mas bajo el entrenamiento es mas lento pero mas preciso.
```
# Define el cargador de datos para el conjunto de prueba en color
  ```python

#Toma el conjunto de datos, lo divide en lotes de 32 ejemplos cada uno y de manera aleatoria se mezclan antes de cada época de entrenamiento.
    train_loader = torch.utils.data.DataLoader(train_dataset_color, batch_size=32, shuffle=True)
```
# Crea el DataLoader para el conjunto de prueba en color
  ```python
#Toma el conjunto de datos, lo divide en lotes de 32 ejemplos cada uno y de manera aleatoria se mezclan antes de cada época de entrenamiento.
    test_loader = torch.utils.data.DataLoader(test_dataset_color, batch_size=32, shuffle=False)
```
# Entrenamiento del modelo en color
  ```python
    num_epochs = 50                      # Número total de épocas de entrenamiento. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            # Verifica si hay una GPU disponible y, si es así, configura el dispositivo de entrenamiento en "cuda" (GPU), de lo contrario, se usa la CPU. 
    model_color.to(device)               # Mueve al modelo al dispositivo seleccionado sea GPU o CPU.

#Listas para almacenar las pérdidas y las precisiones durante el entrenamiento.
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        model_color.train()            # Se llama al modelo de entrenamiento.
        running_loss_color = 0.0       # Variable que lleva el registro de la pérdida acumulada en cada época.

    for images_color, labels_color in train_loader:
        images_color, labels_color = images_color.to(device), labels_color.to(device)

        optimizer_color.zero_grad()          # Se establecen los gradientes de los parámetros del modelo en cero.
 
        outputs_color = model_color(images_color)                 # Se hace una predicción utilizando el modelo en las imágenes de entrada.
        loss_color = criterion(outputs_color, labels_color)       # Se calcula la pérdida entre las predicciones y las etiquetas reales.

        loss_color.backward()
        optimizer_color.step()

        running_loss_color += loss_color.item()

    train_losses.append(running_loss_color / len(train_loader))         # Se calcula el promedio de la pérdida para esa época y se agrega a la lista train_losses.
```

 # Evalúa el modelo en color en el conjunto de prueba
   ```python
    model_color.eval()            # Se pone al modelo en modo de evaluación.
    all_preds_color = []
    all_labels_color = []

    with torch.no_grad():
        for images_color, labels_color in test_loader:
            images_color, labels_color = images_color.to(device), labels_color.to(device)

# Se obtienen las predicciones del modelo y se elige la clase con la puntuación mas alta.
            outputs_color = model_color(images_color)
            _, preds_color = torch.max(outputs_color, 1)
# Las predicciones y las etiquetas reales se agregan a las listas all_preds_color y all_labels_color. 
            all_preds_color.extend(preds_color.cpu().numpy())
            all_labels_color.extend(labels_color.cpu().numpy())

    test_loss_color = criterion(outputs_color, labels_color).item()            # Se calcula la pérdida en el último lote de datos del conjunto de prueba.
    test_acc_color = accuracy_score(all_labels_color, all_preds_color)         # Se calcula la precisión del modelo en el conjunto de prueba utilizando las predicciones y las etiquetas reales. 

    test_losses.append(test_loss_color)
    test_accs.append(test_acc_color)
```
  # Imprime métricas de entrenamiento y prueba en cada época
 ```python
    print(f'Época [{epoch + 1}/{num_epochs}] Pérdida (Color): {train_losses[-1]:.4f}')      # Muestra el número de época actual.
    print(f'Pérdida en Pruebas (Color): {test_losses[-1]:.4f}')          # Muestra la pérdida en el conjunto de pruebas más reciente 
    print(f'Precisión en Pruebas (Color): {test_accs[-1] * 100:.2f}%')   # Muestra la precisión en el conjunto de pruebas más reciente en forma de porcentaje.
```
# Precisión en conjunto de entrenamiento
  ```python
    train_accuracy = accuracy_score(all_labels_color, all_preds_color)             # Calcula la precisión en el conjunto de entrenamiento.
    
    print(f'Precisión en conjunto de entrenamiento (Color): {train_accuracy * 100:.2f}%')
    print(f'Precisión en conjunto de prueba (Color): {test_accs[-1] * 100:.2f}%')
```
# Calcula y muestra la matriz de confusión
  ```python
    conf_matrix_color = confusion_matrix(all_labels_color, all_preds_color)
    class_names = train_dataset_color.classes
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_color, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Color)')
    plt.show()
```
# Resultados de Ejecución

- Precisión en conjunto de entrenamiento (Color): 95.19%
- Precisión en conjunto de prueba (Color): 95.19%

# Interpretación de la Matriz
1.  **Filas:** Cada fila de la matriz representa la clase real a la que pertenecen las muestras.

2.  **Columnas:** Cada columna de la matriz representa la clase a la que el modelo predijo que pertenecen las muestras.

3. **Valores en las Celdas:** Los valores dentro de la matriz indican cuántas muestras se clasificaron de una clase a otra. 


	- **Clase 1 (primera fila):**  tiene una muestra que se clasificó correctamente como clase 1.
	- **Clase 2 (segunda fila):** tiene 43 muestras que se clasificaron correctamente como clase 2 y 5 muestras que se clasificaron incorrectamente como otras clases.
	- **Clase 3 (tercera fila):** tiene 87 muestras que se clasificaron correctamente como clase 3, 1 muestra que se clasificó incorrectamente como clase 4 y 1 muestra que se clasificó incorrectamente como clase 5.
	- **Clase 4 (cuarta fila):**  tiene 44 muestras que se clasificaron correctamente como clase 4 y 1 muestra que se clasificó incorrectamente como clase 5.
	- **Clase 5 (quinta fila):** tiene 442 muestras que se clasificaron correctamente como clase 5, 5 muestras que se clasificaron incorrectamente como clase 3, 1 muestra que se clasificó incorrectamente como clase 4 y 8 muestras que se clasificaron incorrectamente como otras clases.
	- **Clase 6 (sexta fila):**  tiene 15 muestras que se clasificaron correctamente como clase 6 y 4 muestras que se clasificaron incorrectamente como otras clases.
	- **Clase 7 (septima fila):** tiene 113 muestras que se clasificaron correctamente como clase 7 y 1 muestra que se clasificó incorrectamente como clase 4.
	- **Clase 8 (octava fila):** tiene 26 muestras que se clasificaron correctamente como clase 8 y 1 muestra que se clasificó incorrectamente como clase 5.


![Matriz de confusión](https://github.com/cfidrobo/MatrizConfusion/blob/main/95matriz.png)
