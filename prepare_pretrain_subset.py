"""
prepare_pretrain_subset.py

Cura automáticamente un subconjunto mínimo y diverso del dataset K700-2020
para el pre-entrenamiento JEPA.

Inspirado en karpathy/autoresearch: selección basada en métricas para encontrar
el mínimo conjunto efectivo de datos en lugar de usar todo el corpus.

Flujo:
  1. Agrupa las N clases disponibles en `num_clusters` clusters semánticos
     usando TF-IDF + K-Means sobre los nombres de clase.
  2. Elige 1 clase representativa por cluster (más cercana al centroide).
  3. Para cada clase elegida, puntúa los videos por varianza temporal
     (proxy de movimiento/dinamismo) decodificando 3 frames por video.
  4. Selecciona los `videos_per_class` videos más dinámicos por clase.
  5. Escribe el CSV de salida en el formato esperado por VideoDataset
     (sin header, columna única: clase/video.mp4).

Uso:
    python prepare_pretrain_subset.py \\
        --k700_dir /dataset/K700-2020/train \\
        --output_csv pretrain_subset.csv \\
        --num_clusters 30 \\
        --videos_per_class 100 \\
        --diversity_sample 200
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Curación de datos para pre-entrenamiento JEPA")
    parser.add_argument(
        "--k700_dir",
        type=str,
        required=True,
        help="Ruta al directorio train del dataset K700-2020 (contiene subdirectorios por clase)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="pretrain_subset.csv",
        help="Nombre del CSV de salida (se guarda dentro de k700_dir). Default: pretrain_subset.csv",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=30,
        help="Número de clusters semánticos (≈ número de clases seleccionadas). Default: 30",
    )
    parser.add_argument(
        "--videos_per_class",
        type=int,
        default=100,
        help="Máximo de videos a incluir por clase seleccionada. Default: 100",
    )
    parser.add_argument(
        "--diversity_sample",
        type=int,
        default=200,
        help="Cuántos videos muestrear por clase para el scoring de diversidad. Default: 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad. Default: 42",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Paso 1: Descubrir clases disponibles en el directorio
# ---------------------------------------------------------------------------

def discover_classes(k700_dir: str) -> list[str]:
    """Devuelve la lista de clases (subdirectorios con al menos 1 .mp4)."""
    base = Path(k700_dir)
    classes = sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and any(d.glob("*.mp4"))
    )
    return classes


# ---------------------------------------------------------------------------
# Paso 2: Agrupamiento semántico con TF-IDF + K-Means
# ---------------------------------------------------------------------------

def cluster_classes(class_names: list[str], num_clusters: int, seed: int) -> dict[int, list[str]]:
    """
    Agrupa los nombres de clase usando TF-IDF sobre palabras y K-Means.
    Devuelve dict {cluster_id: [class_name, ...]}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # Representar cada nombre de clase como bolsa de palabras TF-IDF
    # (los nombres ya están en inglés con espacios/underscores separando palabras)
    normalized = [name.replace("_", " ").lower() for name in class_names]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(normalized)

    effective_clusters = min(num_clusters, len(class_names))
    km = KMeans(n_clusters=effective_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    clusters: dict[int, list[str]] = {}
    for cls_name, cluster_id in zip(class_names, labels):
        clusters.setdefault(int(cluster_id), []).append(cls_name)

    return clusters, km, vectorizer, X


def select_representative_per_cluster(
    clusters: dict[int, list[str]],
    class_names: list[str],
    km,
    X,
) -> list[str]:
    """
    Para cada cluster, elige la clase más cercana al centroide.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    selected = []

    for cluster_id, members in clusters.items():
        if not members:
            continue
        centroid = km.cluster_centers_[cluster_id].reshape(1, -1)
        member_indices = [class_to_idx[m] for m in members]
        member_vectors = X[member_indices]
        dists = euclidean_distances(member_vectors, centroid).flatten()
        closest_local_idx = int(np.argmin(dists))
        selected.append(members[closest_local_idx])

    return sorted(selected)


# ---------------------------------------------------------------------------
# Paso 3: Scoring de diversidad visual de videos
# ---------------------------------------------------------------------------

def score_video_diversity(video_path: str) -> float:
    """
    Decodifica 3 frames (inicio, medio, fin) y calcula la varianza temporal
    como proxy de movimiento/dinamismo. Retorna -1.0 en caso de error.
    """
    try:
        import torchcodec.decoders as decoders

        decoder = decoders.VideoDecoder(video_path)
        total_frames = decoder.metadata.num_frames
        if total_frames is None or total_frames < 3:
            return 0.0

        indices = [
            0,
            total_frames // 2,
            min(total_frames - 1, total_frames - 1),
        ]
        # Quitar duplicados si el video es muy corto
        indices = sorted(set(indices))

        frames_data = decoder.get_frames_at(indices).data  # (T, C, H, W) uint8
        frames_float = frames_data.float() / 255.0  # normalizar a [0, 1]

        # Varianza entre frames: mide cuánto cambia la imagen a lo largo del tiempo
        variance = float(torch.var(frames_float, dim=0).mean())
        return variance

    except Exception:
        return -1.0


def select_diverse_videos(
    class_name: str,
    k700_dir: str,
    videos_per_class: int,
    diversity_sample: int,
    seed: int,
) -> list[str]:
    """
    Lista los .mp4 de la clase, puntúa hasta `diversity_sample` por varianza
    temporal y devuelve los mejores `videos_per_class` como rutas relativas
    (clase/video.mp4).
    """
    class_dir = Path(k700_dir) / class_name
    all_videos = sorted(class_dir.glob("*.mp4"))

    if not all_videos:
        return []

    # Muestrear para scoring (eficiencia)
    rng = random.Random(seed)
    candidates = all_videos if len(all_videos) <= diversity_sample else rng.sample(all_videos, diversity_sample)

    # Puntuar
    scored = []
    for vp in candidates:
        score = score_video_diversity(str(vp))
        if score >= 0.0:
            scored.append((score, vp))

    if not scored:
        # Fallback: tomar los primeros videos sin scoring
        fallback = all_videos[:videos_per_class]
        return [f"{class_name}/{v.name}" for v in fallback]

    # Ordenar descendente por varianza y tomar top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    top_videos = scored[:videos_per_class]

    return [f"{class_name}/{vp.name}" for _, vp in top_videos]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    k700_dir = args.k700_dir
    if not os.path.isdir(k700_dir):
        print(f"ERROR: No se encontró el directorio: {k700_dir}")
        sys.exit(1)

    # --- Paso 1: Descubrir clases ---
    print(f"\n[1/4] Escaneando clases en: {k700_dir}")
    class_names = discover_classes(k700_dir)
    print(f"      Clases encontradas: {len(class_names)}")

    if len(class_names) == 0:
        print("ERROR: No se encontraron subdirectorios con videos .mp4.")
        sys.exit(1)

    # --- Paso 2: Clustering semántico ---
    num_clusters = min(args.num_clusters, len(class_names))
    print(f"\n[2/4] Agrupando {len(class_names)} clases en {num_clusters} clusters semánticos (TF-IDF + K-Means)...")
    clusters, km, vectorizer, X = cluster_classes(class_names, num_clusters, args.seed)
    selected_classes = select_representative_per_cluster(clusters, class_names, km, X)
    print(f"      Clases seleccionadas ({len(selected_classes)}):")
    for i, cls in enumerate(selected_classes, 1):
        print(f"        {i:2d}. {cls}")

    # --- Paso 3: Scoring de diversidad visual ---
    print(f"\n[3/4] Seleccionando los {args.videos_per_class} videos más dinámicos por clase")
    print(f"      (muestreando hasta {args.diversity_sample} videos por clase para scoring)...")

    all_video_paths: list[str] = []
    for i, cls in enumerate(selected_classes, 1):
        print(f"      [{i:2d}/{len(selected_classes)}] {cls}...", end=" ", flush=True)
        videos = select_diverse_videos(
            cls,
            k700_dir,
            args.videos_per_class,
            args.diversity_sample,
            args.seed,
        )
        all_video_paths.extend(videos)
        print(f"{len(videos)} videos")

    # --- Paso 4: Escribir CSV ---
    # Se usa csv.writer con QUOTE_ALL para que las rutas con espacios o
    # paréntesis (ej. "acting in play/abc.mp4", "backflip (human)/xyz.mp4")
    # queden siempre entre comillas y pandas las lea sin ambigüedad.
    output_path = os.path.join(k700_dir, args.output_csv)
    print(f"\n[4/4] Escribiendo CSV: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for vp in all_video_paths:
            writer.writerow([vp])

    print(f"\n✓ Subconjunto listo:")
    print(f"  Clases: {len(selected_classes)}")
    print(f"  Videos: {len(all_video_paths)}")
    print(f"  CSV:    {output_path}")
    print(f"\nActualiza configs/config.yaml:")
    print(f'  csv_file: "{args.output_csv}"')


if __name__ == "__main__":
    main()
