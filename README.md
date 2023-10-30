# Proyecto 1 módulo 4

Para el presente proyecto se escogió un dataset de la página Kaggle, el cual tiene un contexto interesante, es una compañía que desea lanzar al mercado una nueva marca de teléfonos móviles, para estimar sus precios de venta recopila un conjunto de datos de ventas de teléfonos móviles de diversas compañías que se ofrecen en el mercado.

Despues de la limpieza se identificaron las 4 variable independientes que mantien una correlación positiva con la variable dependiente rango de precios siendo la mas relevante la memoria RAM como independiente

Se utilizaron 2 modelos para el trabajo

**Justificación de la infraestructura seleccionada** : 

El tipo de recurso y el número de nodos influyen en el rendimiento y la escalabilidad. Por ejemplo, en el código se utiliza el recurso "STANDARD_DS12_V2" y se permite un máximo de 6 nodos. Al tener un set de datos de no mas de 2000 filas y 5 columnas no es requerida una capacidad de procesamiento muy grande para el entrenamiento.

```python
cpu_cluster = AmlCompute(
    name=cpu_compute_target,
    type="amlcompute",
    size="STANDARD_DS12_V2",
    min_instances=0,

max_instances=6,

    idle_time_before_scale_down=120,
    tier="Dedicated",
)
```

**Justificación de los parámetros escogidos en el job**: 

Los parámetros del trabajo, como timeout_minutes es un valor establecido por defecto en el programa y se decidió mantener este valor con el mismo y por el volumen de datos se asegura que con un recurso económico bajo se cumplirá con el job, con el objetivo de 5 intentos o max_trials, trial_timeout_minutes que tiene 20 minuto limita a cada prueba a estos mismos, en las pruebas realizadas se obtuvo resultados entre 40 min a 50 min, teniendo holgura por tato en los parámetros escohgidos.

```python
classification_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=5,
    enable_early_termination=True,
)

**Hiperparámetros**

{'my_custom_tag': 'My custom value', 
 'model_explain_run': 'best_run', 
 'pipeline_id_000': '5dfac790c5c209f98a1da2dc1c7fb76f0397324f;c7af0367625be6ac5c2fecbfc72ed444cb7a2111;799d2168db11fc19b9e1c6c1df62f8981ad39fe9;__AutoML_Ensemble__;__AutoML_Stack_Ensemble__', 
 'score_000': '0.9168749999999999;0.915;0.8175000000000001;0.92125;0.9193749999999999', 
 'predicted_cost_000': '0;0;0.5;0;0', 
 'fit_time_000': '0.3878024;0.3851692;0.3878128;4;17', 
 'training_percent_000': '100;100;100;100;100', 
 'iteration_000': '0;1;2;3;4', 
 'run_preprocessor_000': 'MaxAbsScaler;MaxAbsScaler;MaxAbsScaler;;', 
 'run_algorithm_000': 'LightGBM;XGBoostClassifier;ExtremeRandomTrees;VotingEnsemble;StackEnsemble', 
 'automl_best_child_run_id': 'maroon_boot_jb0dh2jnb6_3', 
 'model_explain_best_run_child_id': 'maroon_boot_jb0dh2jnb6_3', 
 'mlflow.rootRunId': 'maroon_boot_jb0dh2jnb6', 
 'mlflow.runName': 'maroon_boot_jb0dh2jnb6', 
 'mlflow.user': 'Boris Bellido'}


Se puede observar que la etiqueta "run_algorithm_000" enumera los nombres de los algoritmos utilizados, como LightGBM, XGBoostClassifier, ExtremeRandomTrees, VotingEnsemble y StackEnsemble. Esto sugiere que se ha realizado un apilamiento (stacking) de modelos, y estos son algunos de los algoritmos utilizados en ese proceso.

Las etiquetas "fit_time_000," "training_percent_000," y "predicted_cost_000" proporcionan información sobre el tiempo de ajuste, el porcentaje de entrenamiento y el costo previsto para cada modelo.






El mejor modelo fue el "Voting" que obtuvo un rendimiento en diferentes aspectos, como la precisión, la recuperación, la puntuación F1 y la capacidad de discriminación. La precisión general del modelo se refleja en la métrica "Precision (Accuracy)," que es del 92.125%. Otras métricas, como el valor F1 y el área bajo la curva (AUC), también indican un buen rendimiento del modelo.


El proyecto se encuentra en:
    
    El video de la explicación  https://1drv.ms/v/s!AnYb6RX9jHJcipkI_50IZ4LUvll4xQ?e=1YTh0S
    El link del github  https://github.com/jboris/ucb_mlc_mod4_trabajo_1
    El link del dataset   https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data


```python

```


```python

```
