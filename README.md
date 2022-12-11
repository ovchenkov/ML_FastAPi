# 1.Предообработка данных:
-Удалены дублирующие строки
-Были обработаны колонки, являющиеся числовыми признаками (убраны ед. изм.)
-Пропуски были только у числовых признаков, которые были заполнены медианой.
-Строчки были приведены к нужным типам данных и 
# 2. Разведоычный анализ данных:
-Были выведены различные количественно-качественные описания исходных данных + построены различные инфографики.
- Чем раньше машина сошла с конвеера (year), тем она дешевле (selling price).
- Чем больше пробег (km_driven), тем цена выше.
- Машины с оптимальным расходом топлива (mileage) (средние значения) - самые дорогие.
- Чем больше объём двигателя (engine), тем дороже машина.
- Чем больше мощность двигателя (max_power), тем машина дороже.
- С ростом числа мест (seats), цена снижается. Семейный сегмент - бюджетный.
- На счёт корреляции признаков. Самая большая корреляция между следующими парами признаков:
- year - km_driven (отрицательная);<br>
- mileage - engine (отрицательная);<br>
- engine - max_power;<br>
- seats - engine;<br>
- Цена на авто максимально коррелирует с мощностью двигателя.
# 3.Модель только на вещественных признаках, лучшие результаты:
- при Lasso-регрессии достигался при alpha = 1990.000, при этом на тесте r2_score = 0.5920 и MSE = 116908346570.9 на тесте
- при ElasticNet-регрессии достигался при alpha = 4950.000, при этом на тесте r2_score = 0.5883 и MSE = 117111484542.2 на тесте
- Качество неудовлетворительное
# 3.Модель с категориальными фичами, лучшие результаты:
- при Lasso-регрессии достигался при alpha = 1990.000, при этом на тесте r2_score = 0.6118 и MSE = 223126896416.7 на тесте
- Качество несколько возрасло
# 4. Feature Engineering:
- Добавил признак power_by_volume - число лошадей на литр: alpha:  20, MSE_train: 94206912638.2, MSE_test: 210853387413.4, r2_score_train: 0.6713, r2_score_test: 0.6332
- Добавил признак squared_year - год изготовления year в квадрате: MSE_train: 94283909833.9, MSE_test: 211502127786.7, r2_score_train: 0.6711, r2_score_test: 0.6321
- Удалил выбросы (km_driven): MSE_train: 93383539171.3, MSE_test: 207319048934.0, r2_score_train: 0.6801, r2_score_test: 0.6393  <------- лучшая модель. Лучше всего подошло добавление признака power_by_volume, остальные методы оказали меньше влияния на улучшение качества.
