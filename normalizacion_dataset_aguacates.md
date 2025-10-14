# Normalización y Limpieza de Datos del Dataset de Aguacates 🥑

## 1️⃣ Revisión y Limpieza Inicial

El dataset fue revisado utilizando `df.info()` y `df.describe()` para detectar valores nulos, tipos de datos y rangos de valores. No se encontraron valores nulos ni duplicados significativos, lo cual permitió continuar con la estandarización de las columnas.

Posteriormente, se realizaron los siguientes pasos de limpieza:

- Eliminación de la columna `Unnamed: 0`, ya que solo representaba un índice sin valor analítico.
- Conversión de la columna `date` a formato de fecha (`datetime`).
- Creación de columnas adicionales derivadas: `date_ts` (timestamp numérico) y `month` (mes del año).
- Renombramiento de las columnas con códigos PLU (`4046`, `4225`, `4770`) por nombres descriptivos:  
  `small_hass_sold`, `large_hass_sold`, `xlarge_hass_sold`.

---

## 2️⃣ Codificación de Variables Categóricas

Se aplicó **codificación dummy** únicamente a la columna `type`, ya que la variable `region` contiene demasiadas categorías, lo que aumentaría innecesariamente la dimensionalidad del dataset.

```python
df = pd.get_dummies(df, columns=['type'], drop_first=True)
```

Esto generó una nueva columna `type_organic`, que toma valores booleanos (`True`/`False`), indicando si el aguacate es orgánico.

---

## 3️⃣ Selección de Variables Numéricas

Se identificaron las variables numéricas que requerían ser llevadas a una misma escala antes de aplicar modelos predictivos. Estas variables están asociadas a precios y volúmenes de ventas, por lo que sus rangos difieren significativamente:

```python
cols_to_scale = [
    'averageprice', 'total_volume', 'small_hass_sold',
    'large_hass_sold', 'xlarge_hass_sold', 'total_bags',
    'small_bags', 'large_bags', 'xlarge_bags'
]
```

Mientras que `averageprice` tiene valores cercanos a 1 o 2 dólares, las variables de volumen como `total_volume` o `small_hass_sold` pueden alcanzar valores de millones de unidades.  
Dejar estas diferencias de escala sin tratar podría causar **sesgos en el modelo**, ya que las variables con valores más grandes dominarían las distancias o gradientes durante el entrenamiento.

---

## 4️⃣ Aplicación del Escalado Estandarizado (Z-Score)

Para la normalización se utilizó el método **Z-Score**, implementado con `StandardScaler` de `scikit-learn`.  
Este método transforma los datos de forma que cada variable tenga una **media de 0** y una **desviación estándar de 1**, según la fórmula:

\(
Z = rac{X - \mu}{\sigma}
\)

Donde:
- \( X \) es el valor original,
- \( \mu \) es la media de la variable,
- \( \sigma \) es la desviación estándar.

Código aplicado:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
```

---

## 5️⃣ Verificación del Resultado

Después de aplicar el escalado, se revisó la estructura del nuevo DataFrame (`df_scaled`) para confirmar que todas las columnas fueron transformadas correctamente y que no se alteraron los tipos de datos no numéricos:

```python
df_scaled.info()
```

Resultado principal:
- Las columnas numéricas presentan valores reescalados (media ≈ 0, desviación estándar ≈ 1).  
- El resto de las variables (`date`, `region`, `type_organic`, `month`, `year`) conservaron su estructura original.  
- No se generaron valores nulos o inconsistentes durante el proceso.

Esto se validó con el comando:

```python
df_scaled.describe()
```

Los resultados mostraron que las variables numéricas tienen:
- Media muy cercana a **0**.  
- Desviación estándar muy cercana a **1**.  
Esto confirma que la normalización se aplicó correctamente.

---

## 6️⃣ Justificación del Método Utilizado

La elección de **Z-Score (StandardScaler)** se justifica por las siguientes razones:

1. **Adecuación para modelos lineales y de distancia**  
   El escalado estandarizado es ideal para algoritmos como regresión lineal, regresión logística, SVM o k-means, que son sensibles a la escala de las variables.

2. **Conserva la distribución original**  
   A diferencia de una normalización min-max, el método Z-score no comprime los valores dentro de un rango fijo (como 0–1), sino que mantiene la forma de la distribución, lo cual es útil si existen colas largas o datos dispersos.

3. **Interpretabilidad**  
   Los valores transformados pueden interpretarse en términos de desviaciones estándar respecto a la media, facilitando la detección posterior de valores atípicos o extremos.

---

## 7️⃣ Conclusiones del Proceso de Normalización

- Se logró **estandarizar las variables numéricas** garantizando comparabilidad entre ellas.  
- Las variables categóricas se manejaron con codificación *dummy*, pero únicamente para la columna `type` (dando origen a `type_organic`), evitando un exceso de variables derivadas de `region`.  
- El conjunto de datos resultante (`df_scaled`) se encuentra limpio, sin duplicados ni valores nulos, y preparado para el **entrenamiento de modelos predictivos de precios**.

En síntesis, el proceso de normalización permitió homogeneizar la escala de los datos, mejorar la calidad estadística del conjunto y sentar las bases para un modelado más preciso y estable.
