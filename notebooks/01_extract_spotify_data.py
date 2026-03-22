# Databricks notebook source

# MAGIC %md
# MAGIC # Spotify Data Extraction (CSV-based)
# MAGIC
# MAGIC Reads exported Spotify playlist CSVs from the `data/` directory, normalizes
# MAGIC them into entity and relationship tables per `graph.yaml`, and writes
# MAGIC Delta tables to Unity Catalog.
# MAGIC
# MAGIC **Data layout:** Each subdirectory under `data/` is a user. Each CSV in
# MAGIC that subdirectory is a playlist. Add more users by creating new subdirectories.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Exported playlist CSVs in `data/<username>/`
# MAGIC - `config.yaml` with `delta` catalog/schema settings
# MAGIC - `graph.yaml` with table schema definitions

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

import yaml
import os
import csv
import hashlib
import base64
from datetime import datetime

# Load config.yaml
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
if not nb_path.startswith("/Workspace"):
    nb_path = "/Workspace" + nb_path
config_path = os.path.join(os.path.dirname(nb_path), "..", "config.yaml")

for candidate in [config_path, "/Workspace/Repos/config.yaml", "config.yaml"]:
    try:
        with open(candidate, "r") as f:
            config = yaml.safe_load(f)
        config_dir = os.path.dirname(candidate)
        print(f"Loaded config from: {candidate}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError("config.yaml not found.")

delta_config = config.get("delta", {})
catalog = delta_config.get("catalog", "spotify_graph")
schema = delta_config.get("schema", "bronze")

# Locate data directory
data_dir = os.path.join(config_dir, "data")
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

user_dirs = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
])

if not user_dirs:
    raise ValueError(f"No user subdirectories found in {data_dir}")

print(f"Delta target: {catalog}.{schema}")
print(f"Data directory: {data_dir}")
print(f"Users found: {user_dirs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Graph Schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType, ArrayType

TYPE_MAP = {
    "STRING": StringType(),
    "INT": IntegerType(),
    "BOOLEAN": BooleanType(),
    "TIMESTAMP": TimestampType(),
    "ARRAY<STRING>": ArrayType(StringType()),
}

graph_candidates = [
    os.path.join(config_dir, "graph.yaml"),
    "/Workspace/Repos/graph.yaml",
    "graph.yaml",
]
for graph_candidate in graph_candidates:
    try:
        with open(graph_candidate, "r") as f:
            graph_config = yaml.safe_load(f)
        print(f"Loaded graph schema from: {graph_candidate}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError("graph.yaml not found.")


def build_schema(table_name):
    for table in graph_config["entities"] + graph_config["relationships"]:
        if table["name"] == table_name:
            return StructType([
                StructField(col["name"], TYPE_MAP[col["type"]], col["nullable"])
                for col in table["columns"]
            ])
    raise ValueError(f"Table '{table_name}' not found in graph.yaml")


print("Schemas loaded:", [t["name"] for t in graph_config["entities"] + graph_config["relationships"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Data from CSVs

# COMMAND ----------


def make_id(*parts):
    """Generate a deterministic short ID by hashing the input parts."""
    raw = "|".join(str(p).strip().lower() for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)[:22].decode("ascii")


def parse_track_uri(uri):
    """Extract track ID from a Spotify URI like 'spotify:track:XXXXX'."""
    if uri and uri.startswith("spotify:track:"):
        return uri.split(":")[-1]
    return None


def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp string to datetime, or return None."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def infer_date_precision(date_str):
    """Infer release_date_precision from the date string format."""
    if not date_str:
        return "year"
    parts = str(date_str).split("-")
    if len(parts) >= 3:
        return "day"
    elif len(parts) == 2:
        return "month"
    return "year"


# Dedup dictionaries for entities
users = {}
playlists = {}
tracks = {}
albums = {}
artists = {}

# Dedup structures for relationship tables
playlist_tracks = {}   # (playlist_id, track_id, position) -> record
track_artists = set()  # (track_id, artist_id)
album_artists = set()  # (album_id, artist_id)

for user_dir_name in user_dirs:
    user_id = user_dir_name
    user_path = os.path.join(data_dir, user_dir_name)

    # Register user
    if user_id not in users:
        users[user_id] = {
            "id": user_id,
            "display_name": user_dir_name.replace("_", " "),
            "external_url": None,
        }

    csv_files = sorted([
        f for f in os.listdir(user_path)
        if f.lower().endswith(".csv")
    ])

    print(f"\nProcessing user: {user_id} ({len(csv_files)} playlists)")

    for csv_file in csv_files:
        playlist_name = os.path.splitext(csv_file)[0]
        # Clean up common export artifacts in filenames
        playlist_name = playlist_name.replace("_", " ").strip()
        # Remove trailing " (1)" style suffixes from duplicate exports
        if playlist_name.endswith(")") and " (" in playlist_name:
            base = playlist_name[:playlist_name.rfind(" (")]
            playlist_name = base

        playlist_id = make_id(user_id, csv_file)
        csv_path = os.path.join(user_path, csv_file)

        # Read CSV
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Register playlist
        if playlist_id not in playlists:
            playlists[playlist_id] = {
                "id": playlist_id,
                "name": playlist_name,
                "description": None,
                "owner_id": user_id,
                "public": None,
                "collaborative": False,
                "snapshot_id": "",
                "image_url": None,
                "total_tracks": len(rows),
            }

        print(f"  Playlist '{playlist_name}': {len(rows)} tracks")

        for position, row in enumerate(rows):
            # --- Track ---
            track_id = parse_track_uri(row.get("Track URI", ""))
            if not track_id:
                continue

            track_name = row.get("Track Name", "")
            album_name = row.get("Album Name", "")
            artist_names_raw = row.get("Artist Name(s)", "")
            artist_names = [a.strip() for a in artist_names_raw.split(";") if a.strip()]
            primary_artist = artist_names[0] if artist_names else "Unknown"

            # Generate deterministic album ID from album name + primary artist
            album_id = make_id(album_name, primary_artist)

            # Parse fields
            duration_ms = int(row.get("Duration (ms)", 0) or 0)
            popularity = int(row.get("Popularity", 0) or 0)
            explicit_raw = row.get("Explicit", "false")
            explicit = explicit_raw.lower() in ("true", "1", "yes")
            release_date = str(row.get("Release Date", ""))
            genres_raw = row.get("Genres", "")
            genres = [g.strip() for g in genres_raw.split(";") if g.strip()] if genres_raw else []

            # Register track (dedup)
            if track_id not in tracks:
                tracks[track_id] = {
                    "id": track_id,
                    "name": track_name,
                    "duration_ms": duration_ms,
                    "explicit": explicit,
                    "disc_number": 1,
                    "track_number": position + 1,
                    "popularity": popularity,
                    "preview_url": None,
                    "album_id": album_id,
                    "external_url": f"https://open.spotify.com/track/{track_id}",
                    "isrc": None,
                }

            # Register album (dedup)
            if album_id not in albums:
                albums[album_id] = {
                    "id": album_id,
                    "name": album_name,
                    "album_type": "album",
                    "release_date": release_date,
                    "release_date_precision": infer_date_precision(release_date),
                    "total_tracks": 0,
                    "external_url": None,
                    "image_url": None,
                }

            # Register artists and relationships
            for artist_name in artist_names:
                artist_id = make_id(artist_name)

                if artist_id not in artists:
                    artists[artist_id] = {
                        "id": artist_id,
                        "name": artist_name,
                        "genres": genres,
                        "popularity": None,
                        "followers": None,
                        "external_url": None,
                        "image_url": None,
                    }

                track_artists.add((track_id, artist_id))

            # Album-artist relationship (primary artist only)
            if artist_names:
                primary_artist_id = make_id(primary_artist)
                album_artists.add((album_id, primary_artist_id))

            # Playlist-track relationship
            key = (playlist_id, track_id, position)
            if key not in playlist_tracks:
                added_by = row.get("Added By", "")
                added_at = row.get("Added At", "")
                playlist_tracks[key] = {
                    "playlist_id": playlist_id,
                    "track_id": track_id,
                    "added_at": parse_timestamp(added_at),
                    "added_by_user_id": added_by if added_by else None,
                    "position": position,
                }

# Update album total_tracks counts
album_track_counts = {}
for t in tracks.values():
    aid = t["album_id"]
    album_track_counts[aid] = album_track_counts.get(aid, 0) + 1
for aid, count in album_track_counts.items():
    if aid in albums:
        albums[aid]["total_tracks"] = count

print(f"\n--- Extraction Summary ---")
print(f"Users:          {len(users)}")
print(f"Playlists:      {len(playlists)}")
print(f"Tracks:         {len(tracks)}")
print(f"Albums:         {len(albums)}")
print(f"Artists:        {len(artists)}")
print(f"playlist_track: {len(playlist_tracks)}")
print(f"track_artist:   {len(track_artists)}")
print(f"album_artist:   {len(album_artists)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"Ensured catalog/schema: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Delta Tables

# COMMAND ----------


def write_table(table_name, records):
    """Create a DataFrame with the graph.yaml schema and write as a Delta table."""
    table_schema = build_schema(table_name)
    df = spark.createDataFrame(records, schema=table_schema)
    full_name = f"{catalog}.{schema}.{table_name}"
    df.write.format("delta").mode("overwrite").saveAsTable(full_name)
    print(f"  {table_name}: {df.count()} rows -> {full_name}")


# --- Entity tables ---

write_table("user", list(users.values()))
write_table("playlist", list(playlists.values()))
write_table("track", list(tracks.values()))
write_table("artist", list(artists.values()))
write_table("album", list(albums.values()))

# --- Relationship tables ---

write_table("playlist_track", list(playlist_tracks.values()))

write_table("track_artist", [
    {"track_id": tid, "artist_id": aid} for tid, aid in track_artists
])

write_table("album_artist", [
    {"album_id": alid, "artist_id": aid} for alid, aid in album_artists
])

print("\nAll Delta tables written successfully.")
