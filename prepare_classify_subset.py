"""
prepare_classify_subset.py

Genera el CSV etiquetado para la fase de clasificación supervisada,
usando el val set de K700-2020 sobre las mismas 30 clases del pre-entrenamiento.

Por qué val y no train:
  - Sin data leakage: el encoder JEPA fue pre-entrenado con videos del train set.
    Usar val garantiza una evaluación honesta.
  - Val está perfectamente balanceado: 48–50 videos por clase, partición oficial.

Formato de salida (compatible con VideoDataset labeled=True):
  video_name,label
  "arm wrestling/abc.mp4",0
  ...

Uso:
    python prepare_classify_subset.py
    python prepare_classify_subset.py \\
        --pretrain_csv /dataset/k700-2020/train/pretrain_subset.csv \\
        --val_dir     /dataset/k700-2020/val \\
        --output_csv  classify_subset.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera CSV etiquetado para clasificación desde el val set de K700-2020"
    )
    parser.add_argument(
        "--pretrain_csv",
        type=str,
        default="/dataset/k700-2020/train/pretrain_subset.csv",
        help="CSV del subset de pre-entrenamiento (para extraer las 30 clases).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/dataset/k700-2020/val",
        help="Directorio raíz del val set de K700-2020.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="classify_subset.csv",
        help="Nombre del CSV de salida (se guarda dentro de val_dir).",
    )
    return parser.parse_args()


def extract_classes_from_pretrain_csv(pretrain_csv: str) -> list[str]:
    """Lee el pretrain_subset.csv y extrae las clases únicas (primera parte del path)."""
    path = Path(pretrain_csv)
    if not path.exists():
        print(f"ERROR: No se encontró {pretrain_csv}")
        sys.exit(1)

    classes = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue
            # Formato: "clase/video.mp4"  →  tomar la parte antes de /
            parts = line.split("/")
            if len(parts) >= 2:
                classes.add(parts[0])

    return sorted(classes)


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)

    if not val_dir.is_dir():
        print(f"ERROR: No se encontró el directorio val: {val_dir}")
        sys.exit(1)

    # --- Paso 1: Extraer las 30 clases del pretrain_subset ---
    print(f"[1/3] Leyendo clases desde: {args.pretrain_csv}")
    classes = extract_classes_from_pretrain_csv(args.pretrain_csv)
    print(f"      Clases encontradas: {len(classes)}")
    for i, cls in enumerate(classes):
        print(f"        {i:2d}. {cls}")

    # --- Paso 2: Recolectar videos del val set para esas clases ---
    print(f"\n[2/3] Recolectando videos desde: {val_dir}")
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    rows: list[tuple[str, int]] = []
    missing_classes = []

    for cls in classes:
        class_dir = val_dir / cls
        if not class_dir.is_dir():
            print(f"  [!] Clase no encontrada en val: '{cls}'")
            missing_classes.append(cls)
            continue

        videos = sorted(class_dir.glob("*.mp4"))
        if not videos:
            print(f"  [!] Sin videos .mp4 en: {class_dir}")
            missing_classes.append(cls)
            continue

        label = label_map[cls]
        for vp in videos:
            rows.append((f"{cls}/{vp.name}", label))

        print(f"      {cls}: {len(videos)} videos (label={label})")

    if missing_classes:
        print(f"\n  ADVERTENCIA: {len(missing_classes)} clases no encontradas en val: {missing_classes}")

    # --- Paso 3: Escribir CSV ---
    output_path = val_dir / args.output_csv
    print(f"\n[3/3] Escribiendo CSV: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["video_name", "label"])  # header requerido por VideoDataset
        for video_name, label in rows:
            writer.writerow([video_name, label])

    # --- Resumen ---
    labels_used = sorted(set(r[1] for r in rows))
    print(f"\n✓ CSV listo:")
    print(f"  Clases:  {len(classes) - len(missing_classes)}")
    print(f"  Videos:  {len(rows)}")
    print(f"  Labels:  {labels_used[0]}–{labels_used[-1]}")
    print(f"  Archivo: {output_path}")
    print(f"\nActualiza configs/config.yaml:")
    print(f'  video_dir: "{val_dir}"')
    print(f'  csv_file_labeled: "{args.output_csv}"')
    print(f"  num_classes: {len(classes) - len(missing_classes)}")


if __name__ == "__main__":
    main()
