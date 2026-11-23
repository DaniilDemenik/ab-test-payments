import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Настройки для красивого вывода графиков
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data(filepath):
    # Загружаем сырые данные и чистим их от аномалий
    print("--- 1. Загрузка и очистка данных ---")
    users = pd.read_excel(filepath, sheet_name='Пользователи')
    payments = pd.read_excel(filepath, sheet_name='Платежи')
    
    initial_users = len(users)
    initial_payments = len(payments)

    # 1. Чистим пользователей
    # Оставляем только валидные группы (A/B), адекватный возраст и непустой город
    users_clean = users[
        users['group'].isin(['A', 'B']) &
        users['age'].between(0, 100) &
        users['city'].notna()
    ].copy()

    # 2. Чистим платежи
    # Убираем записи без user_id
    pay_clean = payments.dropna(subset=['user_id']).copy()
    
    # Джойним с пользователями, чтобы взять их группу
    pay_clean = pay_clean.merge(
        users_clean[['user_id', 'group']], 
        on='user_id', 
        how='inner', 
        suffixes=('_pay', '_user')
    )
    
    # Убираем расхождения в группах между таблицами
    pay_clean = pay_clean[pay_clean['group_pay'] == pay_clean['group_user']]
    pay_clean = pay_clean.rename(columns={'group_pay': 'group'}).drop(columns=['group_user'])

    # Проверяем логику воронки (не может быть шага 4 без шага 3 и т.д.)
    pay_clean = pay_clean[
        ~(pay_clean['step4_success'].notna() & pay_clean['step3_confirmed'].isna()) &
        ~(pay_clean['step3_confirmed'].notna() & pay_clean['step2_entered'].isna()) &
        ~(pay_clean['step2_entered'].notna() & pay_clean['step1_opened'].isna())
    ]

    print(f"Удалено пользователей (аномалии): {initial_users - len(users_clean)}")
    print(f"Удалено платежей (ошибки/несовпадения): {initial_payments - len(pay_clean)}")
    print(f"Осталось в анализе: {len(users_clean)} пользователей, {len(pay_clean)} платежей.\n")
    
    return pay_clean

def calculate_metrics(df):
    # Считаем воронку и конверсии по группам
    print("--- 2. Расчет метрик ---")
    funnel = df.groupby('group').agg(
        attempts=('payment_id', 'count'),
        step1=('step1_opened', lambda x: x.notna().sum()),
        step2=('step2_entered', lambda x: x.notna().sum()),
        step3=('step3_confirmed', lambda x: x.notna().sum()),
        step4=('step4_success', lambda x: x.notna().sum())
    )
    
    # Считаем общую конверсию в успешную оплату (от количества попыток)
    funnel['conversion_rate'] = (funnel['step4'] / funnel['attempts'] * 100).round(2)
    
    print(funnel[['attempts', 'step4', 'conversion_rate']])
    print("\n")
    return funnel

def run_statistical_test(funnel):
    # Проводим односторонний Z-тест для проверки гипотезы
    print("--- 3. Статистическая проверка (Z-тест) ---")
    success_a = funnel.loc['A', 'step4']
    n_a = funnel.loc['A', 'attempts']
    
    success_b = funnel.loc['B', 'step4']
    n_b = funnel.loc['B', 'attempts']
    
    # Проверяем, что конверсия в B больше, чем в A
    stat, pval = proportions_ztest([success_b, success_a], [n_b, n_a], alternative='larger')
    
    print(f"Z-статистика: {stat:.3f}")
    print(f"p-value: {pval:.4f}")
    
    if pval < 0.05:
        print("Вывод: Разница статистически значима (p < 0.05).")
    else:
        print("Вывод: Разница НЕ является статистически значимой.")
    print("\n")

def build_funnel_chart(funnel):
    # Рисуем и сохраняем график воронки в файл
    print("--- 4. Построение графика ---")
    steps = ['step1', 'step2', 'step3', 'step4']
    labels = ['Открыл форму', 'Ввел данные', 'Подтвердил', 'Успешная оплата']
    
    fig, ax = plt.subplots()
    
    for group, color in [('A', '#4C72B0'), ('B', '#DD8452')]:
        # Считаем % прохождения от первого шага
        vals = [funnel.loc[group, s] / funnel.loc[group, 'step1'] * 100 for s in steps]
        ax.plot(labels, vals, marker='o', linewidth=2, label=f'Группа {group}', color=color)
    
    ax.set_title('Воронка оплаты ЖКХ: Сравнение групп A и B', fontsize=14, pad=15)
    ax.set_ylabel('% прохождения от первого шага', fontsize=12)
    ax.set_xlabel('Шаги воронки', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Сохраняем картинку в папку с проектом
    plt.savefig('funnel.png', dpi=200) 
    print("График сохранен в файл 'funnel.png'")
    plt.show()

if __name__ == "__main__":
    # Укажи путь к своему файлу с данными
    file_path = 'Данные для кейса по ПА.xlsx'
    
    # Запускаем по очереди все шаги анализа
    cleaned_data = load_and_clean_data(file_path)
    funnel_metrics = calculate_metrics(cleaned_data)
    run_statistical_test(funnel_metrics)
    build_funnel_chart(funnel_metrics)
    
    print("--- Анализ завершен ---")