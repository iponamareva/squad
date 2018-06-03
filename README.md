# squad


## Описание

Данный проект реализует нейросеть, которая по заданному пользователем контексту и вопросу ищет ответ на этот вопрос в контексте. Нейросеть является адаптированной версией нейросети, описанной в следующей статье: https://arxiv.org/pdf/1704.00051.pdf

## Необходимые пакеты

Для запуска понадобится python3 и несколько дополнительных пакетов.

Для того, чтобы иметь возможность запустить скрипты к этому проекту, необходимо иметь установленный пакет tensorflow. Его установка может оказаться нетривиальной задачей, однако все необходимые инструкции по установке можно найти на официальном сайте Tensorflow: https://www.tensorflow.org/install/

Также необходимы следующие пакеты:
wget, numpy, msgpack, nltk, gensim, fastText, tqdm.

## Обученная модель для скачивания

Для того, чтобы иметь возможность работать с обученной моделью, необходимо в папку со скриптами загрузить следующие файлы:

https://www.dropbox.com/s/fs78ylqddd7tlkh/model_trained_step-1800.data-00000-of-00001?dl=0
https://www.dropbox.com/s/y78c6gki1fazti7/model_trained_step-1800.index?dl=0
https://www.dropbox.com/s/lilpwq2ugs28j35/model_trained_step-1800.meta?dl=0

##  Инструкции по запуску и описание скриптов

0. constants.py содержит константы, используемые в модели (отвечающие за параметры данных для обучения), а так же параметры модели, которые можно настроить: разер батча, количество итераций, размер hidden layer, пути для сохранения и загрузки модели.
1. prepare.py содержит все необходимые для работы с данными функции. 
2. train.py - основной скрипт, который запускает обучение. Пареметры обучения настраиваются в constants.py
3. test.py - скрипт, позволяющий посчитать качество модели на случайном батче исходных тестовых данных.
4. demo.py - скрипт, позволяющий протестировать модель на пользовательских входных данных.

Важно:
Если вы запускаете ЛЮБОЙ скрипт в первый и у вас НЕТ СКАЧАННЫХ ДАННЫХ для обучения, вам необходимо установить в файле CONSTANTS.PY флаг DOWNLOAD в True. Это обеспечивает скачивание необходимых данных в текущую папку при первом запуске скритпа. Предупреждение: скачивание может занять от 15 минут до часа в зависимости от вашего соединения с интернетом.

Запуск TRAIN.PY:
Для обучения необходимо прописать в файле CONSTANTS.PY пути для сохранения вашей модели и запустить TRAIN.PY. Вы можете настроить параметры для обучения (размер батча, количество скрытых слоев) самостоятельно или оставить дефолтные. Обратите внимание, что флаг EP_FLAG устанавливает режим обучения: проход определенного количества эпох либо заданного количества итераций. Рекомендовано задавать количество эпох и устанавливать флаг в True. 

Запуск TEST.PY:
Для запуска вашей модели и проверки на данных тестовой выборки вы можете либо прописать в файле CONSTANTS.PY путь TEST_MODEL_PATH путь к вашей обученной модели, либо оставить это значение дефолтным -- в таком случае будут испоьзоваться параметры, обученные автором.

Запуск DEMO.PY:
Для запуска вашей модели и проверки на пользовательских данных вы можете либо прописать в файле CONSTANTS.PY путь USE_DEMO_MODEL_PATH путь к вашей обученной модели, либо оставить это значение дефолтным -- в таком случае будут испоьзоваться параметры, обученные автором. 

## Результат

На данных, использовавшихся автором, точность на обучающей выборке достиагала 50% на отдельных батчах. На тестовой выборке после 2 эпох результат в среднем был 35%. Для оценки качества использовалась метрика F score.

