# Databricks notebook source

# MAGIC %md
# MAGIC # Load Delta Tables into Neo4j
# MAGIC
# MAGIC Reads entity and relationship Delta tables from Unity Catalog and writes them
# MAGIC to Neo4j as nodes and edges using the Neo4j Spark Connector.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Notebook 01 has been run (Delta tables exist in Unity Catalog)
# MAGIC - Databricks secret scope `neo4j` with keys `uri`, `username`, `password`
# MAGIC - Neo4j Spark Connector JAR installed on the cluster
# MAGIC   (`org.neo4j:neo4j-connector-apache-spark_2.12:5.3.1_for_spark_3`)
# MAGIC - `neo4j` Python driver installed (for creating constraints)

# COMMAND ----------

# MAGIC %pip install neo4j pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

import yaml
import os

# Load config.yaml
config_path = os.path.join(os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().replace("/Workspace", "/Workspace")), "..", "config.yaml")

for candidate in [config_path, "/Workspace/Repos/config.yaml", "config.yaml"]:
    try:
        with open(candidate, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from: {candidate}")
        break
    except FileNotFoundError:
        continue
else:
    raise FileNotFoundError("config.yaml not found.")

delta_config = config.get("delta", {})
neo4j_config = config.get("neo4j", {})

catalog = delta_config.get("catalog", "spotify_graph")
schema = delta_config.get("schema", "bronze")

print(f"Delta source: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Graph Schema

# COMMAND ----------

# Load graph.yaml
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

entity_names = [e["name"] for e in graph_config["entities"]]
relationship_names = [r["name"] for r in graph_config["relationships"]]
print(f"Entities: {entity_names}")
print(f"Relationships: {relationship_names}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Neo4j and Create Uniqueness Constraints

# COMMAND ----------

from neo4j import GraphDatabase

neo4j_uri = dbutils.secrets.get(
    scope=neo4j_config.get("uri_secret_scope", "neo4j"),
    key=neo4j_config.get("uri_secret_key", "uri"),
)
neo4j_username = dbutils.secrets.get(
    scope=neo4j_config.get("username_secret_scope", "neo4j"),
    key=neo4j_config.get("username_secret_key", "username"),
)
neo4j_password = dbutils.secrets.get(
    scope=neo4j_config.get("password_secret_scope", "neo4j"),
    key=neo4j_config.get("password_secret_key", "password"),
)
neo4j_database = neo4j_config.get("database", "neo4j")

# Spark connector options (reused for all writes)
neo4j_options = {
    "url": neo4j_uri,
    "authentication.basic.username": neo4j_username,
    "authentication.basic.password": neo4j_password,
    "database": neo4j_database,
}

# Create uniqueness constraints via the Python driver
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

with driver.session(database=neo4j_database) as session:
    for entity in graph_config["entities"]:
        label = entity["name"].capitalize()
        constraint_name = f"{entity['name']}_id"
        cypher = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        session.run(cypher)
        print(f"  Constraint ensured: {label}.id IS UNIQUE")

driver.close()
print("Neo4j connection verified and constraints created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Nodes
# MAGIC
# MAGIC Load each entity Delta table as Neo4j nodes. Nodes must be created before edges.

# COMMAND ----------

from pyspark.sql.functions import col

for entity in graph_config["entities"]:
    table_name = entity["name"]
    label = table_name.capitalize()
    full_table = f"{catalog}.{schema}.{table_name}"

    print(f"Loading nodes: :{label} from {full_table}")
    df = spark.read.table(full_table)

    df.write.format("org.neo4j.spark.DataSource") \
        .mode("Overwrite") \
        .option("labels", f":{label}") \
        .option("node.keys", "id") \
        .options(**neo4j_options) \
        .save()

    print(f"  :{label} — {df.count()} nodes written")

print("\nAll nodes ingested.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Edges from Relationship Tables
# MAGIC
# MAGIC Read `relationship_type` from `graph.yaml` to dynamically configure each edge write.

# COMMAND ----------

for rel in graph_config["relationships"]:
    rel_type = rel["relationship_type"]
    source_entity = rel_type["source"]
    target_entity = rel_type["target"]
    edge_label = rel_type["name"]

    source_key = f"{source_entity}_id"
    target_key = f"{target_entity}_id"

    # All non-key columns become edge properties
    edge_prop_cols = [
        c["name"] for c in rel["columns"]
        if c["name"] not in (source_key, target_key)
    ]

    full_table = f"{catalog}.{schema}.{rel['name']}"
    print(f"Loading edges: -[:{edge_label}]- from {full_table}")
    print(f"  :{source_entity.capitalize()} -> :{target_entity.capitalize()}", end="")
    if edge_prop_cols:
        print(f"  (properties: {edge_prop_cols})")
    else:
        print()

    rel_df = spark.read.table(full_table) \
        .select(
            col(source_key).alias("source.id"),
            col(target_key).alias("target.id"),
            *[col(c) for c in edge_prop_cols],
        )

    rel_df.write.format("org.neo4j.spark.DataSource") \
        .mode("Overwrite") \
        .option("relationship", edge_label) \
        .option("relationship.save.strategy", "keys") \
        .option("relationship.source.labels", f":{source_entity.capitalize()}") \
        .option("relationship.source.node.keys", "source.id:id") \
        .option("relationship.target.labels", f":{target_entity.capitalize()}") \
        .option("relationship.target.node.keys", "target.id:id") \
        .options(**neo4j_options) \
        .save()

    print(f"  -[:{edge_label}]- — {rel_df.count()} edges written")

print("\nAll relationship-table edges ingested.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Edges from Entity Foreign Keys
# MAGIC
# MAGIC These edges are derived from FK columns on entity tables rather than dedicated
# MAGIC relationship tables.

# COMMAND ----------

# OWNS: User -> Playlist (from playlist.owner_id)
print("Loading edges: -[:OWNS]- from playlist.owner_id")
owns_df = spark.read.table(f"{catalog}.{schema}.playlist") \
    .select(col("owner_id").alias("source.id"), col("id").alias("target.id"))

owns_df.write.format("org.neo4j.spark.DataSource") \
    .mode("Overwrite") \
    .option("relationship", "OWNS") \
    .option("relationship.save.strategy", "keys") \
    .option("relationship.source.labels", ":User") \
    .option("relationship.source.node.keys", "source.id:id") \
    .option("relationship.target.labels", ":Playlist") \
    .option("relationship.target.node.keys", "target.id:id") \
    .options(**neo4j_options) \
    .save()
print(f"  -[:OWNS]- — {owns_df.count()} edges written")

# BELONGS_TO: Track -> Album (from track.album_id)
print("Loading edges: -[:BELONGS_TO]- from track.album_id")
belongs_df = spark.read.table(f"{catalog}.{schema}.track") \
    .select(col("id").alias("source.id"), col("album_id").alias("target.id"))

belongs_df.write.format("org.neo4j.spark.DataSource") \
    .mode("Overwrite") \
    .option("relationship", "BELONGS_TO") \
    .option("relationship.save.strategy", "keys") \
    .option("relationship.source.labels", ":Track") \
    .option("relationship.source.node.keys", "source.id:id") \
    .option("relationship.target.labels", ":Album") \
    .option("relationship.target.node.keys", "target.id:id") \
    .options(**neo4j_options) \
    .save()
print(f"  -[:BELONGS_TO]- — {belongs_df.count()} edges written")

print("\nAll FK-derived edges ingested.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

# Quick count verification via Neo4j Python driver
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

with driver.session(database=neo4j_database) as session:
    print("--- Neo4j Graph Summary ---")

    # Node counts
    for entity in graph_config["entities"]:
        label = entity["name"].capitalize()
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
        count = result.single()["cnt"]
        print(f"  :{label} nodes: {count}")

    # Edge counts
    edge_labels = [r["relationship_type"]["name"] for r in graph_config["relationships"]]
    edge_labels.extend(["OWNS", "BELONGS_TO"])
    for edge_label in edge_labels:
        result = session.run(f"MATCH ()-[r:{edge_label}]->() RETURN count(r) AS cnt")
        count = result.single()["cnt"]
        print(f"  -[:{edge_label}]- edges: {count}")

driver.close()
print("\nNeo4j ingestion complete.")
