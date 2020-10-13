"""
Сценарий --- сущность, объединяющая в себе основные этапы создания рекомендательной системы:

* разбиение данных :ref:`сплиттером <splitters>` на обучающую и валидационную выборки
* подбор гипер-параметров с помощью `optuna <https://optuna.org/>`_
* расчёт :ref:`метрик<metrics>` качества для полученных моделей-кандидатов
* обучение на всём объёме данных с подобранными гипер-параметрами и отгрузка рекомендаций (batch production)

Перед использованием сценария необходимо перевести свои данные во :ref:`внутренний формат <data-preparator>` библиотеки.
"""
from replay.scenarios.main_objective import MainObjective
from replay.scenarios.main_scenario import MainScenario
from replay.scenarios.two_stages_scenario import TwoStagesScenario