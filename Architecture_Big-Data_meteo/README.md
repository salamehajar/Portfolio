# Architecture-Big-Data-meteo - Traitement de Données Météorologiques Temps Réel & Batch

Ce projet déploie un pipeline de données Big Data basé sur l'Architecture Lambda, permettant l'ingestion, le traitement (streaming et batch), le stockage et la visualisation de données météorologiques.

## Arborescence du Projet

L'architecture s'organise en micro-services, séparant l'infrastructure réseau, le calcul, le stockage et la visualisation :

```text
Architecture-Big-Data-meteo/
├── .gitignore
├── README.md
├── cassandra/
│   ├── docker-compose.yml          # Déploiement du nœud Cassandra
│   └── init_cassandra.cql          # Script d'initialisation des schémas (Keyspace & Tables)
├── grafana/
│   ├── docker-compose.yml          # Déploiement de l'interface Grafana
│   ├── Dockerfile                  # Build de l'image (incluant les plugins)
│   └── grafana.db                  # Volume de persistance des Dashboards
├── kafka/
│   └── docker-compose.yml          # Déploiement des brokers Kafka et Zookeeper
├── producer/
│   ├── docker-compose.yml          # Déploiement du conteneur d'ingestion
│   ├── Dockerfile
│   ├── requirements.txt            # Dépendances Python (kafka-python, requests, etc.)
│   └── app/
│       └── producer.py             # Script de récupération de l'API et publication Kafka
└── spark/
    ├── docker-compose.yml          # Déploiement du cluster Spark (Master & Workers)
    ├── Dockerfile
    ├── spark-defaults.conf         # Configuration des packages (Kafka, Cassandra)
    ├── spark-env.sh
    ├── start-spark.sh
    └── apps/
        ├── speedlayer-weather.py   # Job Spark Structured Streaming (Alertes temps réel)
        ├── batchlayer-weather.py   # Job Spark Batch (Agrégations et statistiques)
        └── batch_data/             # Répertoire ignoré par Git (Data Lake Parquet & Checkpoints)
```

## Runbook de Déploiement et d'Exécution
Le déploiement s'effectue de manière séquentielle pour garantir la disponibilité des dépendances réseau (brokers, base de données) avant le lancement des calculs.

1. Démarrage de l'Infrastructure (Docker Compose)
Lancez les services dans cet ordre depuis leurs répertoires respectifs (ou via un réseau unifié si configuré) :

```bash
# Ingestion et Stockage
docker-compose kafka/docker-compose.yml up -d
docker-compose cassandra/docker-compose.yml up -d

# Calcul et Visualisation
docker-compose spark/docker-compose.yml up -d --build
docker-compose grafana/docker-compose.yml up -d --build
```

2. Initialisation des Schémas de Base de Données
Création du Keyspace weather_data et des tables city_stats (historique) et weather_alerts (temps réel).

```bash
# Exécution du script CQL d'initialisation
docker exec -i cassandra1 cqlsh < cassandra/init_cassandra.cql
```
3. Lancement de l'Ingestion (Producer)
Démarrage de la source de données pour alimenter le topic Kafka weather-data.
```bash
docker-compose producer/docker-compose.yml up -d --build
# ou en exécution directe selon la configuration :
# docker exec -it <producer_container> python /app/producer.py
```
4. Exécution de la Speed Layer (Temps Réel)
Traitement des événements Kafka à la volée. Génération d'alertes météo et écriture immédiate dans Cassandra (table weather_alerts) et archivage brut dans le Data Lake (fichiers Parquet).
```bash
docker exec -it spark-master spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,com.datastax.spark:spark-cassandra-connector_2.13:3.5.0 \
    /opt/spark-apps/speedlayer-weather.py
```
Ce processus tourne en continu (Streaming). Normalement, il devrait être possible de le laisser en arrière plan mais comme nous n'avons pas assez de mémoire nous devions arrêter le process plusieurs fois pour la partie de suite.

5. Exécution de la Batch Layer (Historisation)
Traitement des données archivées dans le Data Lake. Calcul des statistiques (moyennes, max) sur des fenêtres temporelles globales et "upsert" dans Cassandra (table city_stats).
```bash
docker exec -it spark-master spark-submit \
    --packages com.datastax.spark:spark-cassandra-connector_2.13:3.5.0 \
    /opt/spark-apps/batchlayer-weather.py
```
6. Visualisation (Grafana)
Accès : http://localhost:3000

Data Source : Apache Cassandra (via le plugin hadesarchitect-cassandra-datasource ou le connecteur officiel selon la version).

Target : cassandra1:9042

Keyspace : weather_data

Les tableaux de bord exploitent des requêtes CQL directes (mode Table) pour afficher les vues servies par les couches Speed et Batch.