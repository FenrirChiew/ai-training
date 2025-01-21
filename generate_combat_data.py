import random as rand

heath_range = [1, 1000]
agi_range = [1, 100]
class_range = [0, 2]
mp_range = [0, 1000]
combat_data = []
while True:
    player_health = rand.randint(heath_range[0], heath_range[1])
    enemy_health = rand.randint(heath_range[0], heath_range[1])
    player_agi = rand.randint(agi_range[0], agi_range[1])
    enemy_agi = rand.randint(agi_range[0], agi_range[1])
    player_class = rand.randint(class_range[0], class_range[1])
    enemy_class = rand.randint(class_range[0], class_range[1])
    mp = rand.randint(mp_range[0], mp_range[1])
    data = [player_health, enemy_health, player_agi, enemy_agi, player_class, enemy_class, mp]
    print('player_health,enemy_health,player_agi,enemy_agi,player_class,enemy_class,mp')
    print(f"{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\t{data[4]}\t{data[5]}\t{data[6]}")

    action = int(input("Enter action (0=attack, 1=defend, 2=retreat, -1 to exit): "))
    if action == -1:
        break

    data.append(action)
    combat_data.append(data)

print("\nAll combat data collected:")
print('player_health,enemy_health,player_agi,enemy_agi,player_class,enemy_class,mp,action')
for data in combat_data:
    print(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]},{data[6]},{data[7]}")
