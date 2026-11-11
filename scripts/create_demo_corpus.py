"""Create a reproducible synthetic Ukrainian IT corpus for offline demonstrations."""

import argparse
from pathlib import Path

TOPICS = [
    ("DNS", "DNS перетворює доменні імена на IP адреси. Перевіряйте кеш і резолвер."),
    ("DHCP", "DHCP автоматично видає IP адресу, шлюз, маску і DNS сервер."),
    ("Linux", "Linux використовує права власника, групи та користувача для файлів."),
    ("Windows Server", "Windows Server керує каталогом Active Directory і службою DNS."),
    ("Git", "Git зберігає історію комітів і підтримує гілки та теги."),
    ("Python", "Python підтримує генератори, ітератори, pathlib і re.finditer."),
    ("Docker", "Docker створює контейнери з образів і керує мережами."),
    ("Cisco", "Cisco комутатори підтримують VLAN, access і trunk порти."),
    ("SSH", "SSH захищає віддалений доступ до Linux серверів."),
    ("Backup", "Резервне копіювання використовує правило 3 2 1 і перевірку відновлення."),
    ("TLS", "TLS захищає HTTPS трафік і перевіряє сертифікати."),
    ("Cloud", "Хмарні сервіси надають віртуальні машини, мережі та сховища."),
    ("Monitoring", "Моніторинг вимірює CPU RAM диск і затримки мережі."),
    ("TCP", "TCP забезпечує доставку пакетів, UDP працює без підтвердження."),
    ("Wireshark", "Wireshark аналізує мережеві пакети та протоколи."),
    ("VirtualBox", "VirtualBox створює віртуальні машини для навчання."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible demo IT notes")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--root", type=Path, default=Path("data/demo_it"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        title, description = TOPICS[i % len(TOPICS)]
        text = [
            f"Нотатка {i + 1:05d}: {title}",
            "Категорія: системне адміністрування. Рік: 2026.",
        ]
        for part in range(12):
            text.append(f"Частина {part + 1}: {description}")
            text.append("Рекомендації: перевірка налаштувань, журналів та документації.")
        (args.root / f"note_{i + 1:05d}.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Generated {args.count} synthetic IT documents in {args.root}")


if __name__ == "__main__":
    main()
