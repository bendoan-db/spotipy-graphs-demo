# Databricks notebook source

# MAGIC %md
# MAGIC # Spotify Data Extraction
# MAGIC
# MAGIC Extracts playlist, track, artist, album, and user data from the Spotify API
# MAGIC and writes it to Delta tables in Unity Catalog.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks secret scope `spotify` with keys `client_id` and `client_secret`
# MAGIC - A Spotify app registered at https://developer.spotify.com/dashboard
# MAGIC - Cluster libraries: `spotipy`, `pyyaml` (installed below)

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

import yaml
import os

# Load config.yaml from the repo root (assumes notebook is run from the repo workspace)
config_path = os.path.join(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().replace("/Workspace", "/Workspace")), "..", "config.yaml")

# Fallback: try common locations
for candidate in [config_path, "/Workspace/Repos/config.yaml", "config.yaml"]:
    try:
        with open(candidate, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {candidate}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError(
        "config.yaml not found. Ensure it is in the repo root or set the path manually."
    )

# Validate required fields
spotify_users = config.get("spotify_users", [])
if not spotify_users:
    raise ValueError("config.yaml: 'spotify_users' must be a non-empty list")

for entry in spotify_users:
    if not entry.get("user_id"):
        raise ValueError("config.yaml: each entry in 'spotify_users' must have a non-empty 'user_id'")

settings = config.get("settings", {})
delta_config = config.get("delta", {})
neo4j_config = config.get("neo4j", {})

playlists_per_user = settings.get("playlists_per_user", 10)
artist_batch_size = settings.get("artist_batch_size", 50)
max_retries = settings.get("max_retries", 3)

catalog = delta_config.get("catalog", "spotify_graph")
schema = delta_config.get("schema", "bronze")

user_ids = [entry["user_id"] for entry in spotify_users]
print(f"Configured users: {user_ids}")
print(f"Playlists per user: {playlists_per_user}")
print(f"Delta target: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Spotify Client

# COMMAND ----------

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = dbutils.secrets.get(scope="spotify", key="client_id")
client_secret = dbutils.secrets.get(scope="spotify", key="client_secret")

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    ),
    requests_timeout=10,
    retries=max_retries,
)

print("Spotify client initialized successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Connection
# MAGIC
# MAGIC Quick check: fetch playlists for the first configured user to confirm auth works.

# COMMAND ----------

test_user_id = user_ids[0]
test_results = sp.user_playlists(test_user_id, limit=5)
test_count = len(test_results.get("items", []))
print(f"Auth check passed. Found {test_count} playlists for user '{test_user_id}'.")

if test_count == 0:
    print(f"WARNING: User '{test_user_id}' has no public playlists. "
          "Verify the user ID is correct and playlists are public.")

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

# Load graph.yaml from the same directory as config.yaml
graph_candidates = [
    os.path.join(os.path.dirname(config_path), "graph.yaml"),
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

print("Graph schemas loaded for:", [t["name"] for t in graph_config["entities"] + graph_config["relationships"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Data from Spotify API

# COMMAND ----------

import time
from spotipy.exceptions import SpotifyException
from datetime import datetime


def api_call_with_backoff(func, *args, **kwargs):
    """Wrap a Spotipy call with retry logic for rate limits and server errors."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = int(e.headers.get("Retry-After", 5))
                print(f"  Rate limited. Retrying in {retry_after + 1}s...")
                time.sleep(retry_after + 1)
            elif e.http_status in (500, 502, 503):
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception(f"API call failed after {max_retries} retries")


# Dedup dictionaries for entities
users = {}
playlists = {}
tracks = {}
artists_stub = {}  # simplified artist data from tracks (id -> record)
albums = {}

# Dedup sets for relationship tables
playlist_tracks = {}   # (playlist_id, track_id, position) -> record
track_artists = set()  # (track_id, artist_id)
album_artists = set()  # (album_id, artist_id)

# Collect artist IDs for batch enrichment
artist_ids_to_fetch = set()

for uid in user_ids:
    print(f"\nProcessing user: {uid}")

    # --- Fetch playlists ---
    result = api_call_with_backoff(sp.user_playlists, uid, limit=playlists_per_user)
    user_playlists = result.get("items", [])

    if not user_playlists:
        print(f"  WARNING: No public playlists found for user '{uid}'. Skipping.")
        continue

    print(f"  Found {len(user_playlists)} playlists")

    for pl in user_playlists:
        # Extract user record from playlist owner
        owner = pl["owner"]
        if owner["id"] not in users:
            users[owner["id"]] = {
                "id": owner["id"],
                "display_name": owner.get("display_name"),
                "external_url": owner.get("external_urls", {}).get("spotify"),
            }

        # Extract playlist record
        if pl["id"] not in playlists:
            playlists[pl["id"]] = {
                "id": pl["id"],
                "name": pl["name"],
                "description": pl.get("description"),
                "owner_id": owner["id"],
                "public": pl.get("public"),
                "collaborative": pl.get("collaborative", False),
                "snapshot_id": pl["snapshot_id"],
                "image_url": pl["images"][0]["url"] if pl.get("images") else None,
                "total_tracks": pl["tracks"]["total"],
            }

        # --- Fetch tracks for this playlist ---
        try:
            items_result = api_call_with_backoff(
                sp.playlist_items, pl["id"], limit=100, additional_types=["track"]
            )
        except SpotifyException as e:
            if e.http_status == 404:
                print(f"  WARNING: Playlist '{pl['id']}' not found (404). Skipping.")
                continue
            raise

        items = items_result.get("items", [])
        while items_result.get("next"):
            items_result = api_call_with_backoff(sp.next, items_result)
            items.extend(items_result.get("items", []))

        print(f"  Playlist '{pl['name']}': {len(items)} items")

        for position, item in enumerate(items):
            track = item.get("track")
            if track is None:
                continue
            if item.get("is_local", False):
                continue
            if not track.get("id"):
                continue

            tid = track["id"]

            # Track record
            if tid not in tracks:
                tracks[tid] = {
                    "id": tid,
                    "name": track["name"],
                    "duration_ms": track["duration_ms"],
                    "explicit": track.get("explicit", False),
                    "disc_number": track.get("disc_number", 1),
                    "track_number": track.get("track_number", 1),
                    "popularity": track.get("popularity"),
                    "preview_url": track.get("preview_url"),
                    "album_id": track["album"]["id"],
                    "external_url": track.get("external_urls", {}).get("spotify", ""),
                    "isrc": track.get("external_ids", {}).get("isrc"),
                }

            # playlist_track relationship
            key = (pl["id"], tid, position)
            if key not in playlist_tracks:
                added_by = item.get("added_by", {})
                playlist_tracks[key] = {
                    "playlist_id": pl["id"],
                    "track_id": tid,
                    "added_at": item.get("added_at"),
                    "added_by_user_id": added_by.get("id") if added_by else None,
                    "position": position,
                }

            # track_artist relationships
            for artist in track.get("artists", []):
                if artist.get("id"):
                    track_artists.add((tid, artist["id"]))
                    artist_ids_to_fetch.add(artist["id"])

            # Album record (from simplified object on the track)
            album = track.get("album", {})
            aid = album.get("id")
            if aid and aid not in albums:
                albums[aid] = {
                    "id": aid,
                    "name": album["name"],
                    "album_type": album.get("album_type", "unknown"),
                    "release_date": album.get("release_date", ""),
                    "release_date_precision": album.get("release_date_precision", ""),
                    "total_tracks": album.get("total_tracks", 0),
                    "external_url": album.get("external_urls", {}).get("spotify", ""),
                    "image_url": album["images"][0]["url"] if album.get("images") else None,
                }

            # album_artist relationships
            for artist in album.get("artists", []):
                if artist.get("id") and aid:
                    album_artists.add((aid, artist["id"]))
                    artist_ids_to_fetch.add(artist["id"])

print(f"\n--- Extraction Summary (before artist enrichment) ---")
print(f"Users:           {len(users)}")
print(f"Playlists:       {len(playlists)}")
print(f"Tracks:          {len(tracks)}")
print(f"Albums:          {len(albums)}")
print(f"Artists to fetch: {len(artist_ids_to_fetch)}")
print(f"playlist_track:  {len(playlist_tracks)}")
print(f"track_artist:    {len(track_artists)}")
print(f"album_artist:    {len(album_artists)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Fetch Artist Details

# COMMAND ----------

artists = {}
artist_id_list = list(artist_ids_to_fetch)

for i in range(0, len(artist_id_list), artist_batch_size):
    batch = artist_id_list[i:i + artist_batch_size]
    result = api_call_with_backoff(sp.artists, batch)

    for artist in result.get("artists", []):
        if artist is None:
            continue
        artists[artist["id"]] = {
            "id": artist["id"],
            "name": artist["name"],
            "genres": artist.get("genres", []),
            "popularity": artist.get("popularity"),
            "followers": artist.get("followers", {}).get("total"),
            "external_url": artist.get("external_urls", {}).get("spotify", ""),
            "image_url": artist["images"][0]["url"] if artist.get("images") else None,
        }

    if (i // artist_batch_size + 1) % 10 == 0:
        print(f"  Fetched {min(i + artist_batch_size, len(artist_id_list))}/{len(artist_id_list)} artists")

print(f"Artists enriched: {len(artists)}")

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

from datetime import datetime


def parse_timestamp(ts_str):
    """Parse ISO 8601 timestamp string to datetime, or return None."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


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

# playlist_track: convert added_at strings to datetime objects
pt_records = []
for rec in playlist_tracks.values():
    pt_records.append({
        **rec,
        "added_at": parse_timestamp(rec["added_at"]),
    })
write_table("playlist_track", pt_records)

write_table("track_artist", [
    {"track_id": tid, "artist_id": aid} for tid, aid in track_artists
])

write_table("album_artist", [
    {"album_id": alid, "artist_id": aid} for alid, aid in album_artists
])

print("\nAll Delta tables written successfully.")
