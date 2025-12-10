import random
import matplotlib.pyplot as plt
import numpy as np

# Коэффициенты целевой функции
a, b, c, d = 20, 3, -40, 1

# Определение функции f(x)
f = lambda x: a + b*x + c*(x**2) + d*(x**3)

# Параметры генетического алгоритма
POP_SIZE = 4            # размер популяции
CROSSOVERS = 2          # число скрещиваний
MUTATIONS = 1           # число мутаций
X_MIN, X_MAX = -10, 53  # границы интервала поиска

# Генерация новой особи (случайный x в диапазоне)
def generate_individual():
    return random.randint(X_MIN, X_MAX)

# Функция приспособленности (fitness)
def fitness(x, mode="max"):
    val = f(x)
    return val if mode == "max" else -val   # инверсия для поиска минимума

# Операция скрещивания (простое арифметическое)
def crossover(parent1, parent2):
    child = (parent1 + parent2) // 2
    return child

# Операция мутации (замена на случайное значение)
def mutate(x):
    return random.randint(X_MIN, X_MAX)

# Один шаг эволюции популяции
def evolve(pop, mode="max"):
    new_pop = pop.copy()

    # Выполняем скрещивания
    for _ in range(CROSSOVERS):
        p1, p2 = random.sample(pop, 2)
        child = crossover(p1, p2)
        new_pop.append(child)

    # Выполняем мутации
    for _ in range(MUTATIONS):
        idx = random.randrange(len(new_pop))
        new_pop[idx] = mutate(new_pop[idx])

    # Селекция: оставляем 4 лучших
    new_pop = sorted(new_pop, key=lambda x: fitness(x, mode), reverse=True)[:POP_SIZE]
    return new_pop

# Запуск ГА и логирование в консоль
def run_ga(mode="max", generations=20):
    pop = [generate_individual() for _ in range(POP_SIZE)]
    history = []

    print(f"=== НАЧАЛО РАБОТЫ ГА ({mode.upper()}) ===")
    print(f"Начальная популяция: {pop}\n")

    for g in range(generations):
        best_x = max(pop, key=lambda x: fitness(x, mode))
        best_val = f(best_x)
        print(f"Поколение {g}: лучший x = {best_x}, f(x) = {best_val}")
        history.append(best_x)

        pop = evolve(pop, mode)
        print(f"Новая популяция: {pop}\n")

    print(f"=== КОНЕЦ РАБОТЫ ГА ({mode.upper()}) ===\n")
    return history

# Визуализация изменения лучшего значения
def visualize(history, title):
    xs = history
    ys = [f(x) for x in xs]

    plt.figure()
    plt.plot(ys, marker='o')
    plt.title(title)
    plt.xlabel('Поколение')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.show()

# Запуск поиска максимума и минимума
max_hist = run_ga(mode="max", generations=20)
min_hist = run_ga(mode="min", generations=20)

visualize(max_hist, "Эволюция поиска максимума")
visualize(min_hist, "Эволюция поиска минимума")
